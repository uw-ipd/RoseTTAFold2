import os
import torch
import util
import glob
#import numpy as np
import gemmi

from pyrosetta import *
init("-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4")

params = {
    "PLDDT_CUT": 0.6, # remove residues below this plddt
    "MINDOM": 50, # min domain size
    "MINGAP": 6, # pae gap for domain splitting
    "MIN_RES_CUT": 3, # do not keep segments shorter than this
}

def setup_docking_mover(counts):
    dock_into_dens = rosetta.protocols.electron_density.DockFragmentsIntoDensityMover()
    dock_into_dens.setB( 16 )
    dock_into_dens.setGridStep( 1 )
    dock_into_dens.setTopN( 500 , 50*counts , 1*counts )
    dock_into_dens.setMinDist( 4 )
    dock_into_dens.setNCyc( 1 )
    dock_into_dens.setClusterRadius( 3 )
    dock_into_dens.setFragDens( 0.9 )
    dock_into_dens.setMinBackbone( False )
    dock_into_dens.setDoRefine( True )
    dock_into_dens.setMaxRotPerTrans( 4 )
    dock_into_dens.setPointRadius( 5 )
    dock_into_dens.setConvoluteSingleR( False )
    dock_into_dens.setLaplacianOffset( 0 )
    return dock_into_dens

def rosetta_density_relax(posein):
    scorefxn = get_fa_scorefxn()
    scorefxn.set_weight( rosetta.core.scoring.elec_dens_fast, 50 )
    scorefxn.set_weight( rosetta.core.scoring.cart_bonded, 0.5 )
    scorefxn.set_weight( rosetta.core.scoring.cart_bonded_angle, 1.0 )
    scorefxn.set_weight( rosetta.core.scoring.pro_close, 0.0 )
    setup = rosetta.protocols.electron_density.SetupForDensityScoringMover()
    relax = rosetta.protocols.relax.FastRelax(scorefxn,1)
    relax.cartesian(True)
    relax.max_iter(100)
    setup.apply(posein)
    relax.apply(posein)

def plddt_trim(model):
    # trim low plddts
    plddt_mask = model['plddt']>params['PLDDT_CUT']
    # remove singletons
    mask,idx,ct = torch.torch.unique_consecutive(plddt_mask,dim=0,return_counts=True,return_inverse=True)
    mask = mask*ct>=params['MIN_RES_CUT']
    plddt_mask = mask[idx]

    pred = model['xyz'][plddt_mask]
    seq = model['seq'][plddt_mask]
    plddt = model['plddt'][plddt_mask]
    pae = model['pae'][plddt_mask][:,plddt_mask]
    L_s = []
    lstart=0
    for li in model['Ls']:
        newl = torch.sum(plddt_mask[lstart:(lstart+li)])
        if newl>0: L_s.append(newl)
        lstart += li
    return {
        'xyz': pred,
        'Ls': L_s,
        'seq': seq,
        'plddt': plddt,
        'pae': pae,
    }

def _pae_split(pae,mindom,mingap):
    L = pae.shape[0]
    bestdelta = mingap
    
    def do_roll(x):
        x = torch.roll(x,(1,1),(0,1))
        x[0,:]=0
        x[:,0]=0
        return x
    
    # 2d cumsum from each corner
    nsum = do_roll(torch.cumsum(torch.cumsum(pae, axis=0), axis=1))
    esum = do_roll(torch.cumsum(torch.cumsum(torch.fliplr(pae), axis=0), axis=1))
    ssum = do_roll(torch.cumsum(torch.cumsum(torch.fliplr(torch.flipud(pae)), axis=0), axis=1))
    wsum = do_roll(torch.cumsum(torch.cumsum(torch.flipud(pae), axis=0), axis=1))
    
    paesum = pae.sum()
    besti,bestj=None,None
    for i in range(L-mindom//2):
        for j in range(i+mindom,L+1):
            jj = L-j
            if (i+jj<mindom):
                continue
            sumin = nsum[i,i]+esum[i,jj]+ssum[jj,jj]+wsum[jj,i]
            sumout = paesum - sumin
            meanin = sumin / (i*i + jj*jj + 2*i*jj)
            meanout = sumout / (L*L - (i*i + jj*jj + 2*i*jj))
            delta = meanout - meanin
            if (delta>bestdelta):
                besti,bestj,bestdelta = i,j,delta

    if besti is None:
        return None
    split_dom = torch.zeros(L).to(torch.bool)
    split_dom[besti:bestj]=1
    return split_dom


def _split_all_by_pae(pae,mindom,mingap):
    L=pae.shape[0]
    mask = torch.ones(L).to(torch.bool)

    domains = [mask]

    #split
    done=False
    while not done:
        new_domains = []
        done=True
        for ds in domains:
            pae_i = pae[ds][:,ds]
            newmask = _pae_split(pae_i, mindom, mingap)
            if newmask is not None:
                done=False
                d1 = ds.clone()
                d1[ds] = newmask
                new_domains.append(d1)
                d2 = ds.clone()
                d2[ds] = ~newmask
                new_domains.append(d2)
            else:
                new_domains.append(ds)
        domains = new_domains

    print ('after split:',[d.sum() for d in domains])

    #join
    done = False
    bestmerge = mingap
    while not done:
        done=True
        
        besti,bestj=None,None
        for i in range(len(domains)-1):
            for j in range(i+1,len(domains)):
                domi = domains[i]
                domj = domains[j]
                pae_i_j = (pae[domi][:,domi].sum() + pae[domj][:,domj].sum())/(torch.square(domi.sum()) + torch.square(domj.sum()))
                pae_ij = (pae[domi][:,domj].sum() + pae[domj][:,domi].sum())/(2*domi.sum()*domj.sum())
                if (pae_ij - pae_i_j < bestmerge):
                    bestmerge,besti,bestj = pae_ij - pae_i_j,i,j
                    done=False

        # merge
        if besti is not None:
            print ('merge',besti,bestj,bestmerge)
            new_domains = [ domains[besti]+domains[bestj] ]
            for i in range(len(domains)):
                if i != besti and i != bestj:
                    new_domains.append(domains[i])
            domains = new_domains
                    
    print ('after merge:',[d.sum() for d in domains])
    return domains


def pae_split(model):
    mindom,mingap = params['MINDOM'],params['MINGAP']
    domains = _split_all_by_pae(model['pae'].to(torch.float), mindom,mingap)

    models = []

    for d in domains:
        pred = model['xyz'][d]
        seq = model['seq'][d]
        plddt = model['plddt'][d]
        pae = model['pae'][d][:,d]
        L_s = []
        lstart=0
        for li in model['Ls']:
            newl = torch.sum(d[lstart:(lstart+li)])
            if newl>0: L_s.append(newl)
            lstart += li

        models.append({
            'xyz': pred,
            'Ls': L_s,
            'seq': seq,
            'plddt': plddt,
            'pae': pae,
        })
    return models

def multidock_model(pdbfile,mapfile, counts):
    pose = pose_from_pdb(pdbfile)
    rosetta.core.scoring.electron_density.getDensityMap(mapfile)
    dock_into_dens = setup_docking_mover(counts)
    dock_into_dens.apply(pose)

    # grab top 'count' poses
    allfiles = glob.glob('EMPTY_JOB_use_jd2_*.pdb')
    allfiles.sort()
    for i,filename in enumerate(allfiles):
        if i==0:
            pose = pose_from_pdb(filename)
        elif i<counts:
            pose.append_pose_by_jump( pose_from_pdb(filename), 1 )
        #os.remove(filename) 
    return pose

def cut_model_from_density(mapfile, pdbfile, mapfileout):
    m = gemmi.read_ccp4_map(mapfile)
    st = gemmi.read_structure(pdbfile)
    offset = gemmi.Position(m.header_float(50),m.header_float(51),m.header_float(52))
    for chain in st[0]:
        for residue in chain:
            for atom in residue:
               atom.pos -= offset
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Constant, 4.0)
    masker.set_to_zero(m.grid, st[0])
    #m.update_ccp4_header()
    m.write_ccp4_map(mapfileout)


def rosetta_density_dock ( preds, mapfile_in ):
    pose = None
    mapfile_working = mapfile_in
    for i,(outfile,model,counts) in enumerate(preds):
        print (outfile,counts)
        model = plddt_trim(model)
        models = pae_split(model)

        # sort by # resolved residues
        models = sorted(models, key=lambda x:sum(x['Ls']), reverse=True)

        for j,m in enumerate(models):
            if (sum(m['Ls'])<10):
                continue
            filename = f"{outfile}.m{i}_d{j}.pdb"
            util.writepdb(filename, m['xyz'], m['seq'], m['Ls'], bfacts=100*m['plddt'])
            pose_i = multidock_model(filename, mapfile_working, counts)
            if pose is None:
                pose = pose_i
            else:
                pose.append_pose_by_jump( pose_i, 1 )

            cut_model_from_density(mapfile_working, filename, f"temp.m{i}_d{j}.mrc")
            mapfile_working = f"temp.m{i}_d{j}.mrc"

    rosetta_density_relax(pose)

    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb(outfile+'.dens.pdb') # overwrite


#rosetta_density_dock('model_00_pred.pdb', 'emd_36027.map')
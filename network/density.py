import os
from pyrosetta import *
init("-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4")

def setup_docking_mover():
    dock_into_dens = rosetta.protocols.electron_density.DockFragmentsIntoDensityMover()
    dock_into_dens.setB( 16 )
    dock_into_dens.setGridStep( 1 )
    dock_into_dens.setTopN( 500 , 5 , 1 )
    dock_into_dens.setMinDist( 3 )
    dock_into_dens.setNCyc( 1 )
    dock_into_dens.setClusterRadius( 3 )
    dock_into_dens.setFragDens( 0.9 )
    dock_into_dens.setMinBackbone( False )
    dock_into_dens.setDoRefine( True )
    dock_into_dens.setMaxRotPerTrans( 10 )
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

def rosetta_density_dock(pdbfile,mapfile):
    pose = pose_from_pdb(pdbfile)
    rosetta.core.scoring.electron_density.getDensityMap(mapfile)
    dock_into_dens = setup_docking_mover()
    dock_into_dens.apply(pose)
    pose = pose_from_pdb('EMPTY_JOB_use_jd2_000001.pdb') # pyrosetta dumps files
    os.remove('EMPTY_JOB_use_jd2_000001.pdb') 
    rosetta_density_relax(pose)

    pose.dump_pdb(pdbfile) # overwrite

#rosetta_density_dock('model_00_pred.pdb', 'emd_36027.map')
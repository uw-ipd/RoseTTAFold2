# RF2
GitHub repo for RoseTTAFold2

Update May 8, 2024:
- Option for density map input

Update Apr 12, 2024:
- Updated config file
- Better memory efficiency during inference
- Symmetry bugfix

## Installation

1. Clone the package
```
git clone https://github.com/uw-ipd/RoseTTAFold2.git
cd RoseTTAFold2
```

2. Create conda environment
```
# create conda environment for RoseTTAFold2
conda env create -f RF2-linux.yml
```
You also need to install NVIDIA's SE(3)-Transformer (**please use SE3Transformer in this repo to install**).
```
conda activate RF2
cd SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ..
```

3. Download pre-trained weights under network directory
```
cd network
wget https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz
tar xvfz RF2_jan24.tgz
cd ..
```

4. Download sequence and structure databases
```
# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates (including *_a3m.ffdata, *_a3m.ffindex)
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```

## Examples
Prepare to run
```
conda activate RF2
cd example
```

### Example: predicting the structure of a monomer in density
Download and unzip emd_36027: https://files.wwpdb.org/pub/emdb/structures/EMD-36027/map/emd_36027.map.gz
```
../run_RF2.sh rcsb_pdb_8J75.fasta -o 8J75 -m emd_36027.map
```

### Expected outputs
Predictions will be output to the folder 1XXX/models/model_final.pdb.  B-factors show the predicted LDDT.
A json file and .npz file give additional accuracy information.

## Additional information
The script `run_RF2.sh` has a few extra options that may be useful for runs:
```
Usage: run_RF2.sh [-o|--outdir name] [-s|--symm symmgroup] [-p|--pair] [-h|--hhpred] input1.fasta ... inputN.fasta
Options:
     -o|--outdir name: Write to this output directory
     -s|--symm symmgroup (BETA): run with the specified spacegroup.
                              Understands Cn, Dn, T, I, O (with n an integer).
     -p|--pair: If more than one chain is provided, pair MSAs based on taxonomy ID.
     -h|--hhpred: Run hhpred to generate templates
     -m|--mapfile: Input electron density map
```

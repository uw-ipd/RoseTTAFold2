# RF2
GitHub repo for RoseTTAFold2

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
```

3. Download pre-trained weights under network directory
```
cd network
wget https://files.ipd.uw.edu/dimaio/RF2_apr23.tgz
tar xvfz RF2_apr23.tgz
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

### Example 1: predicting the structure of a monomer
```
../run_RF2.sh rcsb_pdb_7UGF.fasta -o 7UGF
```

### Example 2: predicting the structure of a heterodimer with paired MSA
```
../run_RF2.sh rcsb_pdb_8HBN.fasta --pair -o 8HBN
```

### Example 3: predicting the structure of a heterotrimer with paired MSA
```
../run_RF2.sh rcsb_pdb_7ZLR.fasta --pair -o 7ZLR
```

### Example 4: predicting the structure of a C6-symmetric homodimer
```
../run_RF2.sh rcsb_pdb_7YTB.fasta --symm C6 -o 7YTB
```

### Example 5: predicting the structure of a C3-symmetric heterodimer (A<sub>3</sub>B<sub>3</sub> complex) with paired MSA
```
../run_RF2.sh rcsb_pdb_7LAW.fasta --symm C3 --pair -o 7LAW
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
```

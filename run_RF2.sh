#!/bin/bash

# make the script stop when error (non-true exit code) occurs
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`
HHDB="$PIPEDIR/pdb100_2021Mar03/pdb100_2021Mar03"

CPU="8"  # number of CPUs to use
MEM="64" # max memory (in GB)

WDIR='rf2out'
#mkdir -p $WDIR/log

conda activate RF2

# process protein (MSA + homology search)
function proteinMSA {
    seqfile=$1
    tag=$2
    hhpred=$3

    # generate MSAs
    if [ ! -s $WDIR/$tag.msa0.a3m ]
    then
        echo "Running HHblits"
        echo " -> Running command: $PIPEDIR/input_prep/make_protein_msa.sh $seqfile $WDIR $tag $CPU $MEM"
        $PIPEDIR/input_prep/make_protein_msa.sh $seqfile $WDIR $tag $CPU $MEM > $WDIR/log/make_msa.$tag.stdout 2> $WDIR/log/make_msa.$tag.stderr
    fi

    if [[ $hhpred -eq 1 ]]
    then
        # search for templates
        if [ ! -s $WDIR/$tag.hhr ]
        then
            echo "Running hhsearch"
            HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $HHDB"
            echo " -> Running command: $HH -i $WDIR/$tag.msa0.ss2.a3m -o $WDIR/$tag.hhr -atab $WDIR/$tag.atab -v 0"
            $HH -i $WDIR/$tag.msa0.a3m -o $WDIR/$tag.hhr -atab $WDIR/$tag.atab -v 0 > $WDIR/log/hhsearch.$tag.stdout 2> $WDIR/log/hhsearch.$tag.stderr
        fi
    fi
}

symm="C1"
pair=0
hhpred=0
fastas=()

## parse command line
USAGESTRING="Usage: $(basename $0) [-o|--outdir name] [-s|--symm symmgroup] [-p|--pair] [-h|--hhpred] input1.fasta ... inputN.fasta"
VALID_ARGS=$(getopt -o o:s:ph --long help,outdir:,symm:,pair,hhpred -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi
eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --help)
        echo $USAGESTRING
        echo "Options:"
        echo "     -o|--outdir name: Write to this output directory"
        echo "     -s|--symm symmgroup (BETA): run with the specified spacegroup."
        echo "                              Understands Cn, Dn, T, I, O (with n an integer)."
        echo "     -p|--pair: If more than one chain is provided, pair MSAs based on taxonomy ID."
        echo "     -h|--hhpred: Run hhpred to generate templates"
        exit 1
        ;;
    -p | --pair)
        pair=1
        shift
        ;;
    -h | --hhpred)
        hhpred=1
        shift
        ;;
    -s | --symm)
        symm="$2"
        shift 2
        ;;
    -o | --outdir)
        WDIR="$2"
        shift 2
        ;;
    --)
        shift; 
        break 
        ;;
  esac
done

mkdir -p $WDIR/log

argstring=""

# split fasta files (if needed)
fastas=()
for i in "$@"
do
    tag=`basename $i | sed -E 's/\.fasta$|\.fas$|\.fa$//'`
    fastas=(${fastas[@]} `awk -v TAG="$tag" -v WDIR="$WDIR" '/^>/{close(out); out=WDIR "/" TAG "_" ++c ".fa"; print out} out!=""{print > out}' $i`)
done

nP=0
for i in "${fastas[@]}"
do
    tag=`basename $i | sed -E 's/\.fasta$|\.fas$|\.fa$//'`

    proteinMSA $i $tag $hhpred
    argstring+="$WDIR/$tag.msa0.a3m"
    if [[ $hhpred -eq 1 ]]; then
        argstring+=":$WDIR/$tag.hhr:$WDIR/$tag.atab"
    fi
    argstring+=" "
    nP=$((nP+1))
done

# Merge MSAs based on taxonomy ID
# to do: add individual chains' hhsearch template results
if [ ${#fastas[@]} -ge 2 ] && [ $pair -eq 1 ]
then
    tag=''
    cmdstr=''
    for i in "${fastas[@]}"
    do
        tagi=`basename $i | sed -E 's/\.fasta$|\.fas$|\.fa$//'`
        tag+=$tagi'.'
        cmdstr+=$WDIR/$tagi.msa0.a3m' '
    done
    echo "Creating merged MSA"
    echo " -> Running command: python $PIPEDIR/input_prep/make_paired_MSA_simple.py $cmdstr > $WDIR/$tag""a3m"
    python $PIPEDIR/input_prep/make_paired_MSA_simple.py $cmdstr > $WDIR/$tag"a3m"
    argstring=$WDIR/$tag"a3m"
fi

# end-to-end prediction
echo "Running RoseTTAFold2 to predict structures"

echo " -> Running command: python $PIPEDIR/network/predict.py -inputs $argstring -prefix $WDIR/models/model -model $PIPEDIR/network/weights/RF2_apr23.pt -db $HHDB -symm $symm"
mkdir -p $WDIR/models

python $PIPEDIR/network/predict.py \
    -inputs $argstring \
    -prefix $WDIR/models/model \
    -model $PIPEDIR/network/weights/RF2_apr23.pt \
    -db $HHDB \
    -symm $symm #2> $WDIR/log/network.stderr 1> $WDIR/log/network.stdout 

echo "Done"

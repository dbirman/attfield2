#!/bin/sh
#SBATCH --job-name=f2bbhv
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig2/bbhv_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig2/bbhv_%a.err
source ~/.bash_profile
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure effects of IT flat gain on raidally distributed receptive fields 
#

N_IMG_PER_CAT=600
LOG="$DATA/runs/fig2/logs/bbhv_$J"


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $LOG.py_out


$py3 $CODE/script/reg_task.py \
    $DATA/runs/fig2/bhv_base_n${N_IMG_PER_CAT}.h5  `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    $N_IMG_PER_CAT                               `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --decoders '(0,5,2)'                         `# Decoder layers` \
    --cuda --batch_size 300 \
    >> $LOG.py_out \
    2> $LOG.py_err
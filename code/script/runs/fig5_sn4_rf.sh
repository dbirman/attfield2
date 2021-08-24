#!/bin/sh
#SBATCH --job-name=f5_sn4_rf
#SBATCH --array=0,1,2,3
#SBATCH --time=2:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:16GB
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig5/logs/sn4_rf_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig5/logs/sn4_rf_%a.err

source ~/.bash_profile
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Run behavioral task with all-layer normalized sensitivity shift attention
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}
N_IMG_PER_CAT=100  # 40 min for 100
LOG="$DATA/runs/fig5/logs/sn4_rf_b$BETA"


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $LOG.py_out

L="[(0,4,0)]"
$py3 $CODE/script/backprop.py \
    $DATA/runs/fig5/sn4_rf_n${N_IMG_PER_CAT}_b$BETA.h5     `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    $N_IMG_PER_CAT                                      `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/sens_norm.py           `# Attention ` \
    --attn_cfg "layer=$L:beta=$BETA"                     \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 100  `# on 16g gpu`                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
     > $LOG.py_out \
    2> $LOG.py_err


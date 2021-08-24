#!/bin/sh
#SBATCH --job-name=f4srf
#SBATCH --array=0,1,2,3
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C=GPU_MEM:32GB
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig4/logs/srf_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig4/logs/srf_%a.err
source ~/.bash_profile
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure effects of stitched gain on raidally distributed receptive fields 
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}
N_IMG_PER_CAT=100 # around 1hr for 100
LOG="$DATA/runs/fig4/logs/srf_b$BETA"


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $DATA/runs/fig4/st-rf_$J.py_out

SPL="gain_split=(0,1,0):splits=[(0,1,1),(0,2,1),(0,3,1)]"
MRG="merges=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/backprop.py \
    $DATA/runs/fig4/rfs_stitch_beta_$BETA.h5            `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    $N_IMG_PER_CAT                                      `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/stitched.py            `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+($BETA-1)*0.35"         \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 9.09 `# = 9 imgs / gig mem`            `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    >> $LOG.py_out \
    2> $LOG.py_err
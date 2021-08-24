#!/bin/sh
#SBATCH --job-name=f2grf
#SBATCH --array=0,1,2,3
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16gb
#SBATCH --constraint=GPU_MEM:32GB
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig2/logs/grf_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig2/logs/grf_%a.err
source ~/.bash_profile
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Run RF tracing with simple gaussian attention
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}
N_IMG_PER_CAT=100 # approx 40 minutes for n_img = 100
LOG="$DATA/runs/fig2/logs/grf_b$BETA"


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $LOG.py_out



$py3 $CODE/script/backprop.py \
    $DATA/runs/fig2/rfs_gauss_beta_$BETA.h5             `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    $N_IMG_PER_CAT                                      `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/gauss_gain.py          `# Model type` \
    --attn_cfg $CODE/proc/att_models/retina_b$BETA.json `# Model params` \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 300                                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    >> $LOG.py_out \
    2> $LOG.py_err
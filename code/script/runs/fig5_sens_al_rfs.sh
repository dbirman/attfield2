#!/bin/sh
#SBATCH --job-name=snsA_rf
#SBATCH --array=0,1,2,3
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig5/snsA-rf_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig5/snsA-rf_%a.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure effects of sentitivity on raidally distributed receptive fields 
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $DATA/runs/fig5/snsA-rf_$J.py_out
L="[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/backprop.py \
    $DATA/runs/fig5/pilot_rfs_sens_al_beta_$BETA.h5     `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    2                                                   `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/sens_shift.py          `# Attention ` \
    --attn_cfg "layer=(0,3,0):beta=$BETA"                \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 3 `# = 1 imgs / gig mem`               `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    >> $DATA/runs/fig5/snsA-rf_$J.py_out \
    2> $DATA/runs/fig5/snsA-rf_$J.py_err






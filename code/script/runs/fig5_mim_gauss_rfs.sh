#!/bin/sh
#SBATCH --job-name=mG_rf
#SBATCH --array=0,1,2,3
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=hns,gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig5/mG-rf_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig5/mG-rf_%a.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure effects of gaussian manual shift on raidally distributed receptive fields 
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $DATA/runs/fig5/mG-rf_$J.py_out
F="$DATA/models/fields/field_gauss_b$BETA.h5"
$py3 $CODE/script/backprop.py \
    $DATA/runs/fig5/pilot_rfs_mim_gauss_beta_$BETA.h5   `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    10                                                  `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/field_shift.py         `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F"    \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 9.09 `# = 9 imgs / gig mem`            `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    >> $DATA/runs/fig5/mG-rf_$J.py_out \
    2> $DATA/runs/fig5/mG-rf_$J.py_err






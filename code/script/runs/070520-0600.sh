#!/bin/sh
#SBATCH --job-name=mFlat
#SBATCH --array=0
#SBATCH --time=7:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/070520/0600_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/070520/0600_%a.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure true RF shifts induced by the flat-field manual shift model
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $DATA/runs/070520/0600_$J.py_out
F="$DATA/models/fields/field_flat_b${BETA}.h5"
$py3 $CODE/script/backprop.py \
    $DATA/runs/070520/rfs_mim_flat_b$BETA.h5          `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5               `# Image Set` \
    100                                                `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                  `# Unit set` \
    '(0,0)'                                            `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/field_shift.py        `# Model type` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F"  `# Model params` \
    --model code/cornet/cornet/cornet_zr.py            `# Model` \
    --abs                                              `# Absolute grads ` \
    --decoders '(0,5,2)'                               `# Decoder layers` \
    --batch_size 9.09 `# = 9 imgs / gig mem`           `# Limit memory` \
    --verbose                                          `# Full Output` \
    --cuda                                             `# Run on GPU` \
    >> $DATA/runs/070520/0600_$J.py_out \
    2> $DATA/runs/070520/0600_$J.py_err





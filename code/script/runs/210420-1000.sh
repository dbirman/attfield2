#!/bin/sh
#SBATCH --job-name=gauss
#SBATCH --array=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/210420/1000_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/210420/1000_%a.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure effects of gaussian gain on receptive fields and behavior
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $DATA/runs/210420/1000_$J.py_out
$py3 $CODE/script/backprop.py \
    $DATA/runs/210420/gauss_beta_$BETA.h5               `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    100                                                 `# Imgs per category` \
    $DATA/models/cZR_100units.csv                       `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/gauss_gain.py          `# Model type` \
    --attn_cfg $CODE/proc/att_models/retina_b$BETA.json `# Model params` \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 300                                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda  --profile                                   `# Run on GPU` \
    >> $DATA/runs/210420/1000_$J.py_out \
    2> $DATA/runs/210420/1000_$J.py_err

#!/bin/sh
#SBATCH --job-name=BaseRF
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/210420/1045_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/210420/1045_%a.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure theoretical receptive fields using the linearized=
# model Cornet_ZLR
#

echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
    >> $DATA/runs/210420/1045.py_out
$py3 $CODE/script/backprop.py \
    $DATA/runs/210420/linear_rfs.h5                     `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    100                                                 `# Imgs per category` \
    $DATA/models/cZR_100units.csv                       `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --model code/cornet/cornet/cornet_zlr.py            `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --regs $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --batch_size 500                                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda  --profile                                   `# Run on GPU` \
    >> $DATA/runs/210420/1045.py_out \
    2> $DATA/runs/210420/1045.py_err
#!/bin/sh
#SBATCH --job-name=BaseRF
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/100520/1155.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/100520/1155.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

echo ">> Starting:" $(date +"%T") >> $DATA/runs/100520/1155.py_out
$py3 $CODE/script/backprop.py \
    $DATA/runs/100520/base_rf-u100-abs.h5               `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    100                                                 `# Imgs per category` \
    $DATA/models/cZR_100units.csv                       `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 500                                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    >> $DATA/runs/100520/1155.py_out \
    2> $DATA/runs/100520/1155.py_err

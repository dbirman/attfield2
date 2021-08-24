#!/bin/sh
#SBATCH --job-name=radBase
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/270420/0917b.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/270420/0917b.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Measure base state of raidally distributed receptive fields 
#

echo ">> Starting:" $(date +"%T")
echo ">> Starting:" $(date +"%T") \
     > $DATA/runs/270420/0917b.py_out
$py3 $CODE/script/backprop.py \
    $DATA/runs/270420/rfs_base.h5                       `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    100                                                 `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --model code/cornet/cornet/cornet_zr.py             `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 300                                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda  --profile                                   `# Run on GPU` \
    >> $DATA/runs/270420/0917b.py_out \
    2> $DATA/runs/270420/0917b.py_err

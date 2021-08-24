#!/bin/sh
#SBATCH --job-name=050320_TLR
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/050320/1230.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/050320/1230.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc
srun $py3 code/script/test_logregs.py \
    $DATA/runs/050320/logregs_test.h5               `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5            `# Image Set` \
    $DATA/models/logregs_iso224_t100_call.npz       `# Regression Path` \
    10                                              `# Num train images` \
    10                                              `# Num test images` \
    "(0,4,2)"                                       `# Decoder layers` \
    --cats bathtub toaster
    --verbose


py3 code/script/test_logregs.py \
    /tmp/logregs_test.h5               `# Output Path` \
    /dev/null            `# Image Set` \
    data/models/logregs_iso224_t100_call.npz       `# Regression Path` \
    10                                              `# Num train images` \
    10                                              `# Num test images` \
    "(0,4,2)"                                       `# Decoder layers` \
    --cats bathtub toaster
    --verbose
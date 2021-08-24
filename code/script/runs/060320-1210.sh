#!/bin/sh
#SBATCH --job-name=ArrTest
#SBATCH --array=0,1
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/060320/1210_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/060320/1210_%a.err
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc
J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0)
BETA=${BETA[$J]}
echo "Running with beta=$BETA"
echo srun $py3 $CODE/script/backprop.py \
    $DATA/runs/060320/arr_test_$J.h5                    `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    30                                                  `# Num images` \
    $DATA/models/cornet_z_2units.csv                    `# Unit set` \
    "(0,0,0)"                                           `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/gauss_gain.py          `# Model type` \
    --attn_cfg $CODE/proc/att_models/retina_b$BETA.json `# Model params` \
    --decoders "(0,4,2)"                                `# Decoder layers` \
    --regs $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --cats banana bathtub                               `# Whitelist` \
    --batch_size 10                                     `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    > $DATA/runs/060320/1210_$J.py_out \
    2> $DATA/runs/060320/1210_$J.py_err

srun $py3 $CODE/script/backprop.py \
    $DATA/runs/060320/arr_test_$J.h5                    `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    30                                                  `# Num images` \
    $DATA/models/cornet_z_2units.csv                    `# Unit set` \
    "(0,0,0)"                                           `# Gradients w.r.t.` \
    --attn $CODE/proc/att_models/gauss_gain.py          `# Model type` \
    --attn_cfg $CODE/proc/att_models/retina_b$BETA.json `# Model params` \
    --decoders "(0,4,2)"                                `# Decoder layers` \
    --regs $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --cats banana bathtub                               `# Whitelist` \
    --batch_size 10                                     `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda                                              `# Run on GPU` \
    > $DATA/runs/060320/1210_$J.py_out \
    2> $DATA/runs/060320/1210_$J.py_err
#!/bin/sh
#SBATCH --job-name=f4senc
#SBATCH --array=0,1,2,3
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig4/logs/senc_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig4/logs/senc_%a.err


source ~/.bash_profile
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Run behavioral task with simple gaussian attention
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}
N_IMG_PER_CAT=100 # approx 1 min / 100
LOG="$DATA/runs/fig4/logs/senc_b$BETA"


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $LOG.py_out

SPL="gain_split=(0,1,0):splits=[(0,1,1),(0,2,1),(0,3,1)]"
MRG="merges=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
IMG=$DATA/imagenet/imagenet_four224l0.h5
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_stitch_b$BETA.h5            `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"         `# Pull layer` \
    --batch_size 100 --cuda                          \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"           `# Image config` \
    --attn $CODE/proc/att_models/stitched.py        `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+($BETA-1)*0.35"     \
    >> $LOG.py_out \
    2> $LOG.py_err

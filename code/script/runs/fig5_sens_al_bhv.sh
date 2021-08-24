#!/bin/sh
#SBATCH --job-name=f5salbhv
#SBATCH --array=0,1,2,3
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:32GB
#SBATCH --output=/scratch/users/kaifox/attfield/data/runs/fig5/logs/salbhv_%a.out
#SBATCH --error=/scratch/users/kaifox/attfield/data/runs/fig5/logs/salbhv_%a.err

source ~/.bash_profile
ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Run behavioral task with layer3 sensitivity shift attention
#

J=$SLURM_ARRAY_TASK_ID
BETA=(1.1 2.0 4.0 11.0)
BETA=${BETA[$J]}
N_IMG_PER_CAT=600  # 15 min for 100
LOG="$DATA/runs/fig5/logs/salbhv_b$BETA"


echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA"
echo ">> Starting:" $(date +"%T") " : J=$J : BETA=$BETA" \
     > $LOG.py_out


ADJ="1+($BETA-1)*0.35"
L="[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/reg_task.py \
    $DATA/runs/fig5/bhv_sens_al_n${N_IMG_PER_CAT}_b$BETA.h5  `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    $N_IMG_PER_CAT                                `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/sens_shift.py`# Attention ` \
    --attn_cfg "layer=$L:beta=$ADJ:neg_mode='fix'"        \
    --decoders '(0,5,2)'                         `# Decoder layers` \
    --cuda --batch_size 5     \
    >> $LOG.py_out \
    2> $LOG.py_err


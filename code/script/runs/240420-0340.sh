ml python/3.6.1
cd $HOME/proj/attfield
source code/script/shrlk.bash_rc

#
# Detection task behavior with wider manual shift
#

for R in 84 112
do
for BETA in 2.0 4.0 11.0
do 
    echo ">> R=$R  BETA=$BETA"
    $py3 $CODE/script/reg_task.py \
        $DATA/runs/240420/rad$R/bhv_shift_b$BETA.h5  `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
        100                                          `# Imgs per category` \
        $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
        --model code/cornet/cornet/cornet_zr.py      `# Model` \
        --attn $CODE/proc/att_models/manual_shift.py   `# Attention model` \
        --attn_cfg $CODE/proc/att_models/rad$R/ret_b$BETA.json  \
        --decoders '(0,5,2)'                        `# Decoder layers` 
done
done

# Flat
for BETA in 2.0 4.0 11.0
do 
    echo ">> BETA=$BETA"
    $py3 $CODE/script/reg_task.py \
        $DATA/runs/240420/flat/bhv_flat_b$BETA.h5    `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
        100                                          `# Imgs per category` \
        $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
        --model code/cornet/cornet/cornet_zr.py      `# Model` \
        --attn $CODE/proc/att_models/flat_gain.py    `# Attention model` \
        --attn_cfg $CODE/proc/att_models/retina_b$BETA.json  \
        --decoders '(0,5,2)'                        `# Decoder layers` 
done

# IT Flat
for BETA in 51.0
do 
    echo ">> BETA=$BETA"
    $py3 $CODE/script/reg_task.py \
        $DATA/runs/240420/flat_it/bhv_flat_b$BETA.h5    `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
        100                                          `# Imgs per category` \
        $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
        --model code/cornet/cornet/cornet_zr.py      `# Model` \
        --attn $CODE/proc/att_models/flat_gain.py    `# Attention model` \
        --attn_cfg $CODE/proc/att_models/it_b$BETA.json  \
        --decoders '(0,5,2)'                        `# Decoder layers` 
done


# Gauss add
for BETA in 2.0 11.0 51.0
do 
    echo ">> BETA=$BETA"
    $py3 $CODE/script/reg_task.py \
        $DATA/runs/240420/gauss_add/bhv_add_b$BETA.h5    `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
        100                                          `# Imgs per category` \
        $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
        --model code/cornet/cornet/cornet_zr.py      `# Model` \
        --attn $CODE/proc/att_models/gauss_add.py    `# Attention model` \
        --attn_cfg $CODE/proc/att_models/retina_b$BETA.json  \
        --decoders '(0,5,2)'                        `# Decoder layers` 
done

$py3 $CODE/script/reg_task.py \
    $DATA/runs/240420/gauss_add/bhv_add_b$BETA.h5    `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    10                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/gauss_add.py    `# Attention model` \
    --attn_cfg $CODE/proc/att_models/retina_b$BETA.json  \
    --decoders '(0,5,2)'    --nodata --cats banana



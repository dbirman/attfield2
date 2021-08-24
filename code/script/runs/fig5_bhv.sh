
# ----------------------------------------------  Shift-Matters Behavior ----

# Gaussian-field imitator behavior
for B in 1.1 2.0 4.0 11.0
do 
#F="$DATA/models/fields/field_gauss_b$B.h5"n # if local
F="$DATA/models/fields/field_gauss_b$B.h5" # if shrlk
$py3 $CODE/script/reg_task.py \
    $DATA/runs/val_rst/bhv_mim_gauss_beta_$B.h5  `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300  --cuda                                  `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/field_shift.py  `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F" \
    --decoders '(0,5,2)'                         `# Decoder layers` 
done

# Flat-field imitator behavior
for B in 1.1 2.0 4.0 11.0
do 
# F="$DATA/models/fields/field_flat_b$B.h5" # if local
F="$DATA/models/fields/field_flat_b$B.h5" # if shrlk
py3 $CODE/script/reg_task.py \
    $DATA/runs/val_rst/bhv_mim_flat_beta_$B.h5   `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300   --cuda                                 `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/field_shift.py  `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F" \
    --decoders '(0,5,2)'                         `# Decoder layers` 
done

# Layer1 Sensitivity gradient behavior
for B in 2.0 4.0
do 
    ADJ="1+($B-1)*0.35"
    $py3 $CODE/script/reg_task.py \
        $DATA/runs/val_rst/bhv_sens_beta_$B.h5       `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
        20                                           `# Imgs per category` \
        $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
        --model code/cornet/cornet/cornet_zr.py      `# Model` \
        --attn $CODE/proc/att_models/sens_shift.py`# Attention ` \
        --attn_cfg "layer=(0,1,0):beta=$ADJ:neg_mode='fix'"        \
        --batch_size 5    \
        --decoders '(0,5,2)'  --cuda                 `# Decoder layers` 
done

# Layer3 Sensitivity gradient behavior
for B in 2.0 4.0
do 
    ADJ="1+($B-1)*0.35"
    $py3 $CODE/script/reg_task.py \
        $DATA/runs/val_rst/bhv_sens_l3_beta_$B.h5    `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
        20                                           `# Imgs per category` \
        $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
        --model code/cornet/cornet/cornet_zr.py      `# Model` \
        --attn $CODE/proc/att_models/sens_shift.py   `# Attention ` \
        --attn_cfg "layer=(0,3,0):beta=$ADJ"            \
        --batch_size 5  --cuda   \
        --decoders '(0,5,2)'                         `# Decoder layers` 
done

## AL-Sensitivity gradient behavior
for B in 11.0
do
L="[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/reg_task.py \
    $DATA/runs/fig5/bhv_sens_al_beta_$B.h5       `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    50  --cuda                                  `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/sens_shift.py   `# Attention ` \
    --attn_cfg "layer=$L:beta=$B:neg_mode='fix'"  \
    --batch_size 3    \
    --decoders '(0,5,2)'                          `# Decoder layers` 
done


for B in 1.1 2.0 4.0 11.0
do 
# scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_mim_gauss_beta_$B.h5 \
#     $DATA/runs/val_rst/bhv_mim_gauss_beta_$B.h5
# scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_mim_flat_beta_$B.h5 \
#     $DATA/runs/val_rst/bhv_mim_flat_beta_$B.h5
scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_sens_beta_$B.h5 \
    $DATA/runs/val_rst/bhv_sens_l1_beta_$B.h5
scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_sens_l3_beta_$B.h5 \
    $DATA/runs/val_rst/bhv_sens_l3_beta_$B.h5
scp $sherlock:$SHRLK_DATA/runs/fig5/bhv_sens_al_beta_$B.h5 \
    $DATA/runs/val_rst/bhv_sens_al_beta_$B.h5
done



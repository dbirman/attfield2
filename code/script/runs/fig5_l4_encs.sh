# Layer4 unit-wise encodings on the FWD task for Shift-Matters models

IMG=$DATA/imagenet/imagenet_four224l0.h5


# Gaussian mimicry encodings
F="$DATA/models/fields/field_gauss_b4.0.h5"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_mim_gauss_b4.0.h5      `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/field_shift.py     `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F" \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`

# Flat mimicry encodings
F="$DATA/models/fields/field_flat_b4.0.h5"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_mim_flat_r2_b4.0.h5       `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/field_shift.py     `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F" \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`

# Layer1 sensitivity shift encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_sens_l1_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --batch_size 5 --cuda                            \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=4.0"              \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`

# Layer3 sensitivity shift encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_sens_l3_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --batch_size 7 --cuda                            \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=(0,3,0):beta=4.0"               \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`

L="[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_sens_al_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --batch_size 7 --cuda                            \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=$L:beta=4.0:neg_mode='fix'"    \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`



# =======================================================================
# ------------------------------------- Pull all layers but no regs  ----

for BETA in 1.1 2.0 4.0 11.0; do
echo "BETA:" $BETA
F="$DATA/models/fields/field_gauss_b$BETA.h5"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig5/lenc_task_mim_gauss_b${BETA}.h5 `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)" --cuda  `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/field_shift.py     `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F"
done

for BETA in 1.1 2.0; do
echo "BETA:" $BETA
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/lenc_task_sens_l3_b$BETA.h5     `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"         `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --batch_size 5 --cuda                            \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=(0,3,0):beta=$BETA"
done 

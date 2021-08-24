# Layer4 unit-wise encodings on the FWD task for Shift-Matters models

IMG=$DATA/imagenet/imagenet_four224l0.h5


# Gaussian mimicry encodings
F="$DATA/models/fields/field_gauss_b4.0.h5"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig5/enc_diag_mim_gauss_b4.0.h5      `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 20  --max_feat 25 --cuda            \
    --attn $CODE/proc/att_models/field_shift.py     `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F"

# Flat mimicry encodings
F="$DATA/models/fields/field_flat_b4.0.h5"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig5/enc_diag_mim_flat_b4.0.h5      `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 20  --max_feat 25 --cuda            \
    --attn $CODE/proc/att_models/field_shift.py     `# Attention` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$F" 

# Layer1 sensitivity shift encodings
ADJ="1+(4.0-1)*0.35"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_diag_sens_l1_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 7  --max_feat 25 --cuda             \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$ADJ"

# Layer3 sensitivity shift encodings
ADJ="1+(4.0-1)*0.35"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_sens_l3_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --batch_size 3  --max_feat 25 --cuda             \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=(0,3,0):beta=$ADJ"

# All-layer sensitivity shift encodings
ADJ="1+(4.0-1)*0.35"
L="[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig5/enc_diag_sens_al_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 3  --max_feat 25 --cuda             \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=$L:beta=$ADJ:neg_mode='fix'"

$py3 $CODE/script/plots/enc_heatmap.py \
    $DATA/runs/fig5/enc_diag.pdf                  `# Output Path` \
    $DATA/runs/fig5/det_n20_meta.pkl              `# Image meta` \
    $DATA/runs/fig2/enc_diag_base_b4.0.h5        `# Encodings` \
    $DATA/runs/fig5/enc_diag_mim_gauss_b4.0.h5     \
    $DATA/runs/fig5/enc_diag_mim_flat_b4.0.h5      \
    $DATA/runs/fig4/enc_diag_sens_l1_b4.0.h5       \
    $DATA/runs/fig5/enc_diag_sens_al_b4.0.h5       \
    --disp "Base" "Mim-Gauss" "Mim-Flat"          `# Names` \
           "Sens-L1" "Sens-AL"           \
    --meta 'lambda img_i, ys, **k: "y: " + str(ys[img_i])' \
    --max_feat 2 --max_img 2 --cmap center
scp $sherlock:$SHRLK_DATA/runs/fig5/enc_diag.pdf $PLOTS/figures/fig5/enc_diag.pdf


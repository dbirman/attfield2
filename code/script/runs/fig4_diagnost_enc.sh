# Layer4 unit-wise encodings on the FWD task for Shift-Matters models

IMG=$DATA/imagenet/imagenet_four224l0.h5


# Flat field attention
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_diag_flat_b4.0.h5           `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 20  --max_feat 25 --cuda            \
    --attn $CODE/proc/att_models/flat_gain.py       `# Attention model` \
    --attn_cfg "layer=(0,1,0):beta=1+(4.0-1)*0.35"

# Stitched convolution model
SPL="gain_split=(0,1,0):splits=[(0,1,1),(0,2,1),(0,3,1)]"
MRG="merges=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_diag_stitch_b4.0.h5         `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 20  --max_feat 25 --cuda            \
    --attn $CODE/proc/att_models/stitched.py        `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+(4.0-1)*0.35" 


$py3 $CODE/script/plots/enc_heatmap.py \
    $DATA/runs/fig4/enc_diag.pdf                  `# Output Path` \
    $DATA/runs/fig5/det_n20_meta.pkl              `# Image meta` \
    $DATA/runs/fig2/enc_diag_base_b4.0.h5         `# Encodings` \
    $DATA/runs/fig4/enc_diag_flat_b4.0.h5          \
    $DATA/runs/fig4/enc_diag_stitch_b4.0.h5        \
    --disp "Base" "Flat" "Stitched"               `# Names` \
    --meta 'lambda img_i, ys, **k: "y: " + str(ys[img_i])' \
    --max_feat 2 --max_img 2
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_diag.pdf $PLOTS/figures/fig4/enc_diag.pdf


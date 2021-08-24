# Layer4 unit-wise encodings on the FWD task for Gain-Matters models

IMG=$DATA/imagenet/imagenet_four224l0.h5

# Flat-field IT encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_flat_b4.0.h5        `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/flat_gain.py       `# Attention model` \
    --attn_cfg "layer=(0,1,0):beta=1+(4.0-1)*0.35" \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`

# Stitched-field IT encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_stitch_b4.0.h5         `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/stitched.py        `# Attention model` \
    --attn_cfg "split=(0,1,0):merge=(0,4,0):beta=1+(4.0-1)*0.35"  \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`

SPL="gain_split=(0,1,0):splits=[(0,1,1),(0,2,1),(0,3,1)]"
MRG="merges=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_stitch_b4.0.h5         `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)' --cuda                                `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/stitched.py        `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+(4.0-1)*0.35"  \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`


for B in 1.1; do
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_flat_b4.0.h5 $DATA/runs/fig4/enc_task_flat_b4.0.h5
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_stitch_b4.0.h5 $DATA/runs/fig4/enc_task_stitch_b4.0.h5
done

# =======================================================================
# ------------------------------------- Pull all layers but no regs  ----

SPL="gain_split=(0,1,0):splits=[(0,1,1),(0,2,1),(0,3,1)]"
MRG="merges=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
for BETA in 1.1 2.0 4.0 11.0; do
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig4/enc_task_stitch_b$BETA.h5       `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)" --cuda  `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/stitched.py        `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+($BETA-1)*0.35"
done




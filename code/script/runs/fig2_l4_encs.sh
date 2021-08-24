IMG=$DATA/imagenet/imagenet_four224l0.h5

# Base IT encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig2/lenc_task_base.h5               `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"         `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --cuda

# Gaussian-field IT encodings
for BETA in 1.1 2.0 4.0 11.0; do
echo "BETA:" $BETA
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig2/lenc_task_gauss_b${BETA}.h5     `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"         `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/cts_gauss_gain.py  `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$BETA"            \
    --cuda
done

scp $sherlock:$SHRLK_DATA/runs/fig2/lenc_task_base.h5 $DATA/runs/fig2/lenc_task_base.h5
for BETA in 2.0 4.0; do
scp $sherlock:$SHRLK_DATA/runs/fig2/lenc_task_gauss_b$BETA.h5 $DATA/runs/fig2/lenc_task_gauss_b$BETA.h5
done
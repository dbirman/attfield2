Genration of similar data for shrinkage model

```bash
IMG=ssddata/imagenet/imagenet_four224l0.h5
N_IMG_PER_CAT=100
BETA=0.2
py3 $CODE/script/encodings.py \
    $DATA/runs/fig5/lenc_ms_n${N_IMG_PER_CAT}_b$BETA.h5   `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    "(0,4,0)"         `# Pull layer` \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"           `# Image config` \
    --batch_size 10      \
    # or --batch_size 10 --cuda
    --attn $CODE/proc/att_models/manual_shrink.py       `# Model type` \
    --attn_cfg "layer=(0,4,0):beta=$BETA"               `# Model params`
```
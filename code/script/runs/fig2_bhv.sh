# -----------------------------------------------  Base & Gaussian Behavior ----

$py3 $CODE/script/reg_task.py \
    $DATA/runs/val_rst/bhv_base300.h5            `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --decoders '(0,5,2)'              `# Decoder layers`




















for B in 1.1 2.0 4.0 11.0
do 
$py3 $CODE/script/reg_task.py \
    $DATA/runs/val_rst/bhv_gauss_beta_$B.h5      `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/cts_gauss_gain.py`# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$B"             \
    --decoders '(0,5,2)'                         `# Decoder layers` 
done




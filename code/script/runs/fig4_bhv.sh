
# -----------------------------------------------  Gain-Matters Behavior ----

# Flat-field attention behavior
for B in 4.0 11.0
do 
$py3 $CODE/script/reg_task.py \
    $DATA/runs/val_rst/bhv_flat_beta_$B.h5       `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/flat_gain.py    `# Attention model` \
    --attn_cfg "layer=(0,1,0):beta=1+($B-1)*0.35" \
    --decoders '(0,5,2)'                         `# Decoder layers` 
done

# Stitched gain attention behavior
SPL="gain_split=(0,1,0):splits=[(0,1,1),(0,2,1),(0,3,1)]"
MRG="merges=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]"
for B in 1.1 2.0 4.0 11.0
do 
$py3 $CODE/script/reg_task.py \
    $DATA/runs/val_rst/bhv_stitch_beta_$B.h5     `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300 --cuda                                   `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --attn $CODE/proc/att_models/stitched.py     `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+($B-1)*0.35"  \
    --decoders '(0,5,2)'                         `# Decoder layers` 
done




for B in 1.1 2.0 4.0 11.0; do
# scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_flat_beta_$B.h5 $DATA/runs/val_rst/bhv_flat_beta_$B.h5
scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_stitch_beta_$B.h5 $DATA/runs/val_rst/bhv_stitch_beta_$B.h5
done
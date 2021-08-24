# Layer4 unit-wise encodings on the FWD task for Shift-Matters models

# Base
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig2/enc_diag_base_b4.0.h5           `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 20  --max_feat 25 --cuda 

# Gaussian
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig2/enc_diag_gauss_b4.0.h5          `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'         `# Pull layer` \
    --gen_cfg "img=$IMG:n=20"                       `# Image config` \
    --batch_size 20  --max_feat 25 --cuda            \
    --attn $CODE/proc/att_models/cts_gauss_gain.py  `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=4.0"


$py3 $CODE/script/plots/enc_heatmap.py \
    $DATA/runs/fig2/enc_diag.pdf                  `# Output Path` \
    $DATA/runs/fig5/det_n20_meta.pkl              `# Image meta` \
    $DATA/runs/fig2/enc_diag_base_b4.0.h5         `# Encodings` \
    $DATA/runs/fig2/enc_diag_gauss_b4.0.h5         \
    --disp "Base" "Gaussian"                      `# Names` \
    --meta 'lambda img_i, ys, **k: "y: " + str(ys[img_i])' \
    --max_feat 2 --max_img 2 --cmap center
scp $sherlock:$SHRLK_DATA/runs/fig2/enc_diag.pdf $PLOTS/figures/fig2/enc_diag.pdf


# Edge detector encodings
for B in 1.1 2.0 4.0 11.0
do
$py3 $CODE/script/encodings.py \
    $DATA/runs/val_rst/enc_edge_gauss_b$B.h5        `# Output Path` \
    $CODE/proc/image_gen/bars.py                    `# Image Set` \
    $CODE/proc/nets/edges.py                        `# Model` \
    '(0,1)'                                         `# Pull layers` \
    --attn $CODE/proc/att_models/cts_gauss_gain.py  `# Attn type` \
    --attn_cfg "layer=(0,0):beta=$B"                `# Attn params` \
    --gen_cfg "ns=[5, 10, 20]:size=112"             `# Image config`
done
# Visualize edge detector encodings
$py3 $CODE/script/plots/enc_heatmap.py \
    $PLOTS/runs/val_rst/enc_edge_gauss.pdf        `# Output Path` \
    $DATA/runs/050520/bars_meta.pkl               `# Image meta` \
    $DATA/runs/val_rst/enc_edge_gauss_b1.1.h5     `# Encodings` \
    $DATA/runs/val_rst/enc_edge_gauss_b2.0.h5      \
    $DATA/runs/val_rst/enc_edge_gauss_b4.0.h5      \
    $DATA/runs/val_rst/enc_edge_gauss_b11.0.h5     \
    --disp 1.1 2.0 4.0 11.0     
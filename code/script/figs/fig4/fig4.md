# Attfield Figure 4

> __Fig 4: Gain matters model__
> 
> (a) Flat gain and stitched gain model diagrams / encoding examples
> (b) These don’t shift RFs according to the gaussian shift model
> - `fig4/quiver`
> - `fig4/line_shift_{}`
> - `fig4/line_size_{}`
> - Flat disrupts the pattern of RF shifts, while stitched a) detaches shift from the edge units b) detaches gain parameter from shift
> (c) These show primarily the “gain matters” hypothesis in feature space
> - `fig4/flex_axis_{}`
> - `fig4/info_point_AUC_{}`
> (d) These match behavior of the gaussian gain
> - `fig4/btv_beta_gain`





## (a) Diagrams / encoding examples

[1] Diagrams:

```bash
# All we need here is a blank axis to annotate in illustrator
```

Encoding examples:

```bash

```

## (b) No corresponding RF shift

Quiver plots:

```bash

# -------------------------------------------------  Data Gen ----

# Run stitch backprop and summary
sbatch code/script/runs/fig4_stitch_rfs.sh
for BETA in 1.1 2.0 4.0 11.0
do
$py3 code/script/summarize_rfs.py \
    $DATA/runs/fig4/summ_stitch_b${BETA}_ell.csv   `# Output Path` \
    $DATA/runs/fig4/pilot_rfs_stitch_beta_$BETA.h5 `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv              `# Unit set` \
    $CODE/proc/rf_models/gaussian.py               `# RF Model`
done

# Run flat summary script
for BETA in 1.1 2.0 4.0 11.0
do
$py3 code/script/summarize_rfs.py \
    $DATA/runs/fig4/summ_flat_b${BETA}_ell.csv     `# Output Path` \
    $DATA/runs/280420/rfs_flat_beta_$BETA.h5       `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv              `# Unit set` \
    $CODE/proc/rf_models/gaussian.py               `# RF Model`
done

# Download
for BETA in 1.1 2.0 4.0 11.0
do
#scp $sherlock:$SHRLK_DATA/runs/fig4/summ_stitch_b${BETA}_ell.csv $DATA/runs/fig4/summ_stitch_b${BETA}_ell.csv
scp $sherlock:$SHRLK_DATA/runs/fig4/summ_flat_b${BETA}_ell.csv $DATA/runs/fig4/summ_flat_b${BETA}_ell.csv
done

for F in $(cat ~/tmp/files_list.txt)
do
scp $DATA/$F $sherlock:$SHRLK_DATA/$F 
done

# ----------------------------------------------------  Plotting ----

# Plot quivers with size map
for LAYER in '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'; do echo $LAYER;
for BETA in 1.1 2.0 4.0 11.0
do
py3 code/script/plots/sizemap_quiver.py \
    $PLOTS/figures/fig4/quiver/stitch_b${BETA}_l${LAYER}.pdf`# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv            `# Base RFs ` \
    $DATA/runs/fig4/summ_stitch_b${BETA}_ell.csv   `# Cued RFs ` \
    $LAYER --lim 224 --em 2.1415                   `# Misc`
done; done
for LAYER in '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'; do echo $LAYER;
for BETA in 1.1 2.0 4.0 11.0
do
py3 code/script/plots/sizemap_quiver.py \
    $PLOTS/figures/fig4/quiver/flat_b${BETA}_l${LAYER}.pdf  `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv            `# Base RFs ` \
    $DATA/runs/fig4/summ_flat_b${BETA}_ell.csv     `# Cued RFs ` \
    $LAYER --lim 224 --em 2.1415                   `# Misc`
done; done

# ------------------------------------------------- Size line plot
for MODEL in stitch; do echo $MODEL;
$py3 code/script/plots/rf_size_dists.py \
    $PLOTS/figures/fig4/line_size_${MODEL}.pdf        `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig4/summ_${MODEL}_b1.1_ell.csv        `# Cued RFs` \
    $DATA/runs/fig4/summ_${MODEL}_b2.0_ell.csv  \
    $DATA/runs/fig4/summ_${MODEL}_b4.0_ell.csv  \
    $DATA/runs/fig4/summ_${MODEL}_b11.0_ell.csv \
    --compare \
    $DATA/runs/270420/summ_cts_gauss_b1.1_ell.csv    `# Gaussian RFs` \
    $DATA/runs/270420/summ_cts_gauss_b2.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b4.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --ylim -20 15 --xlim 0 180 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --line_span 30
done

# only stitch model: for manuscript
$py3 code/script/plots/rf_size_dists.py \
    $PLOTS/figures/fig4/line_size_stitch.pdf          `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig4/summ_stitch_b1.1_ell.csv          `# Cued RFs` \
    $DATA/runs/fig4/summ_stitch_b2.0_ell.csv  \
    $DATA/runs/fig4/summ_stitch_b4.0_ell.csv  \
    $DATA/runs/fig4/summ_stitch_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --figsize 5 5 --ylim -1.1 0.2\
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv

# ------------------------------------------------- Shift line plot

for MODEL in stitch; do echo $MODEL;
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig4/line_shift_${MODEL}.pdf      `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv              `# Base RFs` \
    $DATA/runs/fig4/summ_${MODEL}_b1.1_ell.csv       `# Cued RFs` \
    $DATA/runs/fig4/summ_${MODEL}_b2.0_ell.csv  \
    $DATA/runs/fig4/summ_${MODEL}_b4.0_ell.csv  \
    $DATA/runs/fig4/summ_${MODEL}_b11.0_ell.csv \
    --compare \
    $DATA/runs/270420/summ_cts_gauss_b1.1_ell.csv    `# Gaussian RFs` \
    $DATA/runs/270420/summ_cts_gauss_b2.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b4.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --ylim -20 15 --xlim 0 180 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --line_span 30
done

# only stitch model : for manuscript
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig4/line_shift_stitch.pdf        `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv              `# Base RFs` \
    $DATA/runs/fig4/summ_stitch_b1.1_ell.csv         `# Cued RFs` \
    $DATA/runs/fig4/summ_stitch_b2.0_ell.csv  \
    $DATA/runs/fig4/summ_stitch_b4.0_ell.csv  \
    $DATA/runs/fig4/summ_stitch_b11.0_ell.csv \
    --loc 56 56 --px_per_degree 22 --rad 22 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --figsize 5 5 --ylim -1.05 .82 \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --norm_summ $DATA/runs/270420/summ_base_ell.csv \
    --norm_param ">np.sqrt(n.major_sigma)"

# ------------------------------------------------- Gain line plot



$py3 code/script/plots/act_dists.py \
    $PLOTS/figures/fig4/line_gain_stitch.pdf          `# Output Path` \
    $DATA/runs/fig2/lenc_task_base.h5                 `# Base Acts` \
    $DATA/runs/fig4/enc_stitch_b1.1.h5          `# Cued RFs` \
    $DATA/runs/fig4/enc_stitch_b2.0.h5  \
    $DATA/runs/fig4/enc_stitch_b4.0.h5  \
    $DATA/runs/fig4/enc_stitch_b11.0.h5 \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --figsize 5 5 --n_img 3 --no_read
    # if changed anything about gain calculation: --no_read
for BETA in 1.1 2.0 4.0 11.0; do
scp $sherlock:$SHRLK_DATA/runs/fig2/lenc_task_gauss_b${BETA}.h5.sgain.npz $DATA/runs/fig2/lenc_task_gauss_b${BETA}.h5.sgain.npz
done
scp $sherlock:$SHRLK_DATA/runs/fig2/lenc_task_base.h5.sd.npz $DATA/runs/fig2/lenc_task_base.h5.sd.npz

```



## (c) “Gain matters” hypothesis in feature space

```bash
# Download data from sherlock
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_flat_b4.0.h5 $DATA/runs/fig4/enc_task_flat_b4.0.h5
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_stitch_b4.0.h5 $DATA/runs/fig4/enc_task_stitch_b4.0.h5

for MODEL in 'flat' 'stitch'
do echo $MODEL;
py3 code/script/plots/flex_readout.py \
    $PLOTS/figures/fig2/read_gauss.pdf          `# Out `\
    $DATA/models/logregs_iso224_t100.npz         `# Regs`\
    $DATA/runs/fig2/fnenc_task_base.h5           `# Base`\
    $DATA/runs/fig2/enc_task_gauss_b4.0.h5    `# Cued`\
    '(0,4,3)' --jtr 0.15 --em 2.1415             `# Misc`
done

for MODEL in 'flat' 'stitch'
do echo $MODEL;
py3 code/script/plots/flex_readout.py \
    $PLOTS/figures/fig4/flex_{}_$MODEL.pdf          `# Out `\
    $DATA/models/logregs_iso224_t100.npz         `# Regs`\
    $DATA/runs/fig2/fnenc_task_base.h5           `# Base`\
    $DATA/runs/fig4/enc_task_${MODEL}_b4.0.h5    `# Cued`\
    '(0,4,3)' --jtr 0.15 --em 2.1415             `# Misc`
done
```


## (d) Match behavior of gaussian gain

```bash
py3 $CODE/script/plots/bhv_by_beta.py \
    $PLOTS/figures/fig4/bhv_flat_beta_gain.pdf    `# Output Path` \
    $DATA/runs/val_rst/bhv_flat_beta_1.1.h5       `# Gaussian` \
    $DATA/runs/val_rst/bhv_flat_beta_2.0.h5        \
    $DATA/runs/val_rst/bhv_flat_beta_4.0.h5        \
    $DATA/runs/val_rst/bhv_flat_beta_11.0.h5       \
    --disp "1.1" "2.0" "4.0" "11.0"              `# Display names` \
    --cond Flat  Flat  Flat  Flat                  \
    --cmp $DATA/runs/val_rst/bhv_base.h5          `# Uncued control` \
    --cmp_disp Dist                               `# Control name` \
    --y_rng 0.5 1.0                               `# Standard scale` \
    --bar1 0.69 --bar2 0.87

py3 $CODE/script/plots/bhv_by_beta.py \
    $PLOTS/figures/fig4/bhv_stitch_beta_gain.pdf   `# Output Path` \
    $DATA/runs/val_rst/bhv_flat_beta_11.0.h5       \
    $DATA/runs/val_rst/bhv_stitch_beta_1.1.h5     `# Gaussian` \
    $DATA/runs/val_rst/bhv_stitch_beta_2.0.h5      \
    $DATA/runs/val_rst/bhv_stitch_beta_4.0.h5      \
    $DATA/runs/val_rst/bhv_stitch_beta_11.0.h5     \
    --disp "1.1" "2.0" "4.0" "11.0"              `# Display names` \
    --cond Stitched  Stitched  Stitched  Stitched  \
    --cmp $DATA/runs/val_rst/bhv_base.h5          `# Uncued control` \
    --cmp_disp Dist                               `# Control name` \
    --y_rng 0.5 1.0                               `# Standard scale` \
    --bar1 0.69 --bar2 0.87
```



## TODO

__(a)__ 

__(b,c,e)__ Center units for all, any?

__(c)__ What units to select? Resize input images?

__(d)__ Only the human-matched strength?









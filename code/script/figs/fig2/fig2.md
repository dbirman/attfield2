# Attfield Figure 2

> __Fig 2: Attention also having this other qualitative effect of shifting RFs__
> 
> (a) Receptive field model
> - Selection of receptive fields _Should explain this anyway and it will stop people worrying about the gridding in quiver plots._
> (b) Effect of gain on RFs (with specific RF examples) replicates expected phenomenology: shift and shrink.
> - `fig2/quiver/`
> - `fig2/line_shift_gauss`
> - `fig2/line_size_gauss`
> (c) Dual effect on stimulus representation
> - The attended location becomes more task salient: `fig2/flex_axis_gauss`
> - The edges of the attended location become more task salient: `fig2/info_point_AUC_gauss`


## (a) Network

## (a) Receptive field model

Diagram of receptive field model on top of true receptive fields

```bash
## Compute gaussian receptive field params
sbatch code/script/runs/270420-0917b.sh # Base Uncued
$py3 code/script/summarize_rfs.py \
    $DATA/runs/270420/summ_base_ell.csv        `# Output Path` \
    $DATA/runs/270420/rfs_base.h5              `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv          `# Unit set` \
    $CODE/proc/rf_models/gaussian.py           `# RF Model`

## Plot a receptive field with parametric model overlay
code/script/figs/rf_ellipse.py
```

Distribution of RF statistics

```bash
py3 code/script/plots/rf_stats.py \
    $PLOTS/figures/fig2/rf_stats_ell_base.pdf \
    $DATA/runs/270420/summ_base_ell.csv \
    --bins 20  \
    --custom "maj_rad=lambda row: np.sqrt(row.major_sigma)" \
             "min_rad=lambda row: np.sqrt(row.minor_sigma)" \
             "true_area=lambda row: np.sqrt(row.major_sigma*row.minor_sigma)*np.pi"
```


## (b) Effect of gain on RFs

##### [1] Visualization of change: RF heatmaps

```bash
## Compute gaussian receptive field params under gaussian attention
for BETA in 1.1 2.0 4.0 11.0; do
$py3 code/script/summarize_rfs.py \
    $DATA/runs/270420/summ_cts_gauss_b${BETA}_ell.csv `# Output Path` \
    $DATA/runs/270420/rfs_cts_gauss_beta_${BETA}.h5   `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv                 `# Unit set` \
    $CODE/proc/rf_models/gaussian.py                  `# RF Model`
done
for BETA in 1.1 2.0 4.0 11.0; do
scp $sherlock:$SHRLK_DATA/runs/270420/summ_cts_gauss_b${BETA}_ell.csv $DATA/runs/270420/summ_cts_gauss_b${BETA}_ell.csv
done
scp $sherlock:$SHRLK_DATA/runs/270420/rfs_cts_gauss_beta_11.0.h5 $DATA/runs/270420/rfs_cts_gauss_beta_11.0.h5
scp $sherlock:$SHRLK_DATA/runs/270420/summ_cts_gauss_b11.0_ell.csv $DATA/runs/270420/summ_cts_gauss_b11.0_ell.csv

## This script generates both the difference and overlay maps
code/script/figs/rf_diffmap.py
```

##### [2] Statistics of change: Quiver plot and line plot

```bash
cp $PLOTS/runs/270420/visual_cts_gauss_b4.0.pdf $PLOTS/figures/fig2/quiver_gauss_b4.0.pdf 
cp $PLOTS/runs/270420/line_cts_gauss.pdf $PLOTS/figures/fig2/line_gauss.pdf

# quiver plot with size change
for LAYER in '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'; do echo $LAYER;
for BETA in 1.1 2.0 4.0 11.0; do
py3 code/script/plots/sizemap_quiver.py \
    $PLOTS/figures/fig2/quiver/quiv_gausss_b${BETA}_l${LAYER}.pdf `# Output` \
    $DATA/runs/270420/summ_base_ell.csv                 `# Base RFs ` \
    $DATA/runs/270420/summ_cts_gauss_b${BETA}_ell.csv   `# Cued RFs ` \
    $LAYER --lim 224 --em 2.1415                        `# Misc`
done; done

# ------------------------------------------------- Size line plot
$py3 code/script/plots/rf_size_dists.py \
    $PLOTS/figures/fig2/line_size_gauss.pdf           `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/270420/summ_cts_gauss_b1.1_ell.csv     `# Cued RFs` \
    $DATA/runs/270420/summ_cts_gauss_b2.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b4.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --figsize 5 5 --ylim -.32 .1\
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv


# ------------------------------------------------- Shift line plot
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig2/line_shift_gauss.pdf          `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/270420/summ_cts_gauss_b1.1_ell.csv     `# Cued RFs` \
    $DATA/runs/270420/summ_cts_gauss_b2.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b4.0_ell.csv  \
    $DATA/runs/270420/summ_cts_gauss_b11.0_ell.csv \
    --loc 56 56 --px_per_degree 22 --rad 22 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --figsize 5 5 --ylim -.05 .7 \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --norm_summ $DATA/runs/270420/summ_base_ell.csv \
    --norm_param ">np.sqrt(n.major_sigma)"


# ------------------------------------------------- Gain line plot

$py3 code/script/plots/act_dists.py \
    $PLOTS/figures/fig2/line_gain_gauss_ylim.pdf      `# Output Path` \
    $DATA/runs/fig2/lenc_task_base.h5                 `# Base Acts` \
    $DATA/runs/fig2/lenc_task_gauss_b1.1.h5           `# Cued RFs` \
    $DATA/runs/fig2/lenc_task_gauss_b2.0.h5  \
    $DATA/runs/fig2/lenc_task_gauss_b4.0.h5  \
    $DATA/runs/fig2/lenc_task_gauss_b11.0.h5 \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --raw_ylim 0.2 1e6 --sd_ylim .2 15 \
    --figsize 5 5 --n_img 3 --no_read 
    # if changed anything about gain calculation: --no_read
scp  $sherlock:/scratch/users/kaifox/attfield/plots/figures/fig2/line_gain_gauss_ylim.pdf plots/figures/fig2/line_gain_gauss_ylim.pdf

# ------------------------------------------------- Quiver Sizemap
# quiver plot with size change
for BETA in 1.1 2.0 4.0 11.0; do
py3 code/script/plots/rf_size_dists.py \
    $PLOTS/figures/fig2/line/line_gauss_b${BETA}.pdf    `# Output` \
    $DATA/runs/270420/summ_base_ell.csv                 `# Base RFs ` \
    $DATA/runs/270420/summ_cts_gauss_b${BETA}_ell.csv   `# Cued RFs ` \
    $LAYER --lim 224 --em 2.1415                        `# Misc`
done


# ------------------------------------------------- Gain line comparison

$py3 code/script/plots/act_dists.py \
    $PLOTS/figures/fig2/line_gain_gauss_compare_zoom.pdf      `# Output Path` \
    $DATA/runs/fig2/lenc_task_base.h5                 `# Base Acts` \
    $DATA/runs/fig2/lenc_task_gauss_b1.1.h5           `# Cued RFs` \
    $DATA/runs/fig2/lenc_task_gauss_b2.0.h5  \
    $DATA/runs/fig2/lenc_task_gauss_b4.0.h5  \
    $DATA/runs/fig2/lenc_task_gauss_b11.0.h5 \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --sd_ylim 0 6 \
    --n_bins 100 --bootstrap_n 50 \
    --figsize 5 5 --n_img 3 --is_comparison --no_raw
    # if changed anything about gain calculation: --no_read
scp  $sherlock:/scratch/users/kaifox/attfield/plots/figures/fig2/line_gain_gauss_compare_zoom.pdf plots/figures/fig2/line_gain_gauss_compare_zoom.pdf
```


## (c) Dual effect on stimulus representation

```bash
## Produce the spatialized regression outputs for task images
IMG=$DATA/imagenet/imagenet_four224l0.h5
# Base IT encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig2/fnenc_task_base.h5              `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --regs $DATA/models/logregs_iso224_t100.npz --cuda    `# Regressions`
# Gaussian-field IT encodings
$py3 $CODE/script/encodings.py \
    $DATA/runs/fig2/enc_task_gauss_b2.0.h5          `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    code/cornet/cornet/cornet_zr.py                 `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/proc/att_models/cts_gauss_gain.py  `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=2.0"              \
    --regs $DATA/models/logregs_iso224_t100.npz  --cuda   `# Regressions`
# Download data
scp $sherlock:$SHRLK_DATA/runs/fig2/fnenc_task_base.h5 $DATA/runs/fig2/fnenc_task_base.h5
scp $sherlock:$SHRLK_DATA/runs/fig2/enc_task_gauss_b2.0.h5 $DATA/runs/fig2/enc_task_gauss_b2.0.h5

# Flex readout plot
py3 code/script/plots/flex_readout.py \
    $PLOTS/figures/fig2/flex_{}_gauss.pdf        `# Out `\
    $DATA/models/logregs_iso224_t100.npz         `# Regs`\
    $DATA/runs/fig2/fnenc_task_base.h5           `# Base`\
    $DATA/runs/fig2/enc_task_gauss_b4.0.h5       `# Cued`\
    '(0,4,3)' --jtr 0.15 --em 2.1415             `# Misc`
    
```

and this effect on stimulus representation yields a behavioral shift

```bash
py3 $CODE/script/plots/bhv_by_beta.py \
   $PLOTS/figures/fig2/bhv_beta_gain_n600.pdf     `# Output Path` \
   $DATA/runs/fig2/bhv_gauss_n600_beta_1.1.h5      `# Gaussian` \
   $DATA/runs/fig2/bhv_gauss_n600_beta_2.0.h5       \
   $DATA/runs/fig2/bhv_gauss_n600_beta_4.0.h5       \
   $DATA/runs/fig2/bhv_gauss_n600_beta_11.0.h5      \
   --disp "1.1" "2.0" "4.0" "11.0"               `# Display names` \
   --cond Gauss  Gauss  Gauss  Gauss              \
   --cmp $DATA/runs/fig2/bhv_base_n600.h5        `# Uncued control` \
   --cmp_disp Dist                               `# Control name` \
   --y_rng 0.5 1.0                               `# Standard scale` \
   --bar1 0.687 --bar2 0.776
scp $sherlock:$SHRLK_PLOTS/figures/fig2/bhv_beta_gain_n600.pdf plots/figures/fig2/bhv_beta_gain_n600.pdf
```



## TODO

__(2a)__ How to treat classifier layer. Prototype / logistic regression or FC layer? Average pooling vs shared weights?

__(a)__ Choose one without border bias.

__(3b)__
Show multiple? Show at multiple layers? (compare to Baruch Yeshurun fig 1)
Get real ellipses in there.

__(3c)__
Update aesthetics of quiver and size of both plots.
Add highlights of selected units

__(3e)__
Add multi-layer propagation plots.






## Debugging Behavioral Test

Conclusion: something went wrong with the logreg file on sherlock, reuploaded and seems to be working agsain.

```bash
N_IMG_PER_CAT=50
$py3 $CODE/script/reg_task.py \
    $DATA/debug/local_bhv_base.h5                `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    $N_IMG_PER_CAT                               `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --decoders '(0,5,2)'                         `# Decoder layers`

$py3 $CODE/script/reg_task.py \
    $DATA/debug/shrlk-logregs_bhv_base.h5        `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    $N_IMG_PER_CAT                               `# Imgs per category` \
    $DATA/debug/shrlk_logregs_iso224_t100.npz    `# Regressions` \
    --model code/cornet/cornet/cornet_zr.py      `# Model` \
    --decoders '(0,5,2)'                         `# Decoder layers`

py3 $CODE/script/plots/bhv_by_beta.py \
   $DATA/debug/bhv-debug.pdf     `# Output Path` \
   $DATA/debug/local_bhv_base.h5      `# Gaussian` \
   $DATA/debug/shrlk-logregs_bhv_base.h5       \
   --disp "local" "logregs"
```














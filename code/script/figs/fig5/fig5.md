# Attfield Figure 5

> __Fig 5: Shift matters model__
> 
> (a) Mimicry and sens-shift model diagrams / encoding examples
> (b) These DO shift RFs according to the models they try to replicate
> (c) These show primarily the “shift matters” hypothesis in feature space
> (d) These match behavior of the gaussian gain

(a) Mimicry and sens-shift model diagrams / encoding examples
(b) These DO shift RFs according to the models they try to replicate

- `fig5/quiver`
- `fig5/line_shift_{}`
- `fig5/line_size_{}`
- _Do we need mim-flat?_
- _Use sens-l3._

(c) These show primarily the “shift matters” hypothesis in feature space

- `fig5/flex_axis_{}`
- `fig5/info_point_AUC_{}`

(d) These match behavior of the gaussian gain

- `fig5/btv_beta_gain`






<!-- ====================================================================== -->
<!-- ====================================================================== -->
## (a) Diagrams / encoding examples
<!-- ====================================================================== -->
<!-- ====================================================================== -->

[1] Diagrams:

```bash
code/script/figs/fig5/diagram_plots.py
# -> plots/figures/fig5/sens_shift_diagram.pdf
```

Encoding examples:

```bash
# ---------------------- Image metadata
IMG=$DATA/imagenet/imagenet_four224l0.h5
py3 $CODE/script/get_image_meta.py \
    $DATA/runs/fig5/det_n20_meta.pkl       `# Output Path` \
    $CODE/proc/image_gen/det_task.py       `# Image Set` \
    --gen_cfg "img=$IMG:n=20"              `# Image config`


# ---------------------  Sensitivity gradient attention  ----
for B in 1.1 2.0 4.0 11.0
do
$py3 $CODE/script/encodings.py \
    $DATA/runs/val_rst/enc_edge_sens_b$B.h5        `# Output Path` \
    $CODE/proc/image_gen/bars.py                    `# Image Set` \
    $CODE/proc/nets/edges.py                        `# Model` \
    '(0,1)'                                         `# Pull layers` \
    --attn $CODE/proc/att_models/sens_shift.py      `# Attention ` \
    --attn_cfg "layer=(0,1):beta=$B:neg_mode='fix'"         \
    --gen_cfg "ns=[5, 10, 20]:size=112"             `# Image config`
done

$py3 $CODE/script/plots/enc_heatmap.py \
    $PLOTS/runs/val_rst/enc_edge_sens.pdf        `# Output Path` \
    $DATA/runs/050520/bars_meta.pkl              `# Image meta` \
    $DATA/runs/val_rst/enc_edge_sens_b1.1.h5     `# Encodings` \
    $DATA/runs/val_rst/enc_edge_sens_b2.0.h5      \
    $DATA/runs/val_rst/enc_edge_sens_b4.0.h5      \
    $DATA/runs/val_rst/enc_edge_sens_b11.0.h5     \
    --disp 1.1 2.0 4.0 11.0                      `# Display names`
```







<!-- ====================================================================== -->
<!-- ====================================================================== -->
## (b) No corresponding RF shift
<!-- ====================================================================== -->
<!-- ====================================================================== -->


```bash
# -------------------------------------------------------- RF Diagnostic ----
sbatch code/script/runs/fig5_sens_l1_rfs.sh
py3 code/script/sanity/rf_diagnostic.py \
    $DATA/runs/270420/rfs_base.pdf     `# Output Path` \
    $DATA/runs/270420/rfs_base.h5      `# Backprop data`
scp $sherlock:$SHRLK_DATA/runs/fig5/sna_rf_n100_b4.0.h5 $DATA/runs/fig5/sna_rf_n100_b4.0.h5

# -------------------------------------------------------- RF Summaries ----

# for B in 1.1 2.0 4.0 11.0
# do
# $py3 code/script/gen_unit_field.py \
#     $DATA/runs/270420/field_gauss_b${B}.h5        `# Output Path` \
#     $DATA/runs/270420/summ_cts_gauss_b${B}_com.csv  `# Attended CoMs` \
#     $DATA/runs/270420/summ_base_ell.csv           `# Base CoMs` \
#     "(0, 4, 0)"                                   `# Layer to mimic` \
#     224                                           `# Input space size`
# done
for BETA in 1.1 2.0 4.0 11.0; do
    scp data/runs/270420/field_gauss_b${BETA}.h5 $sherlock:$SHRLK_DATA/models/fields/field_gauss_b${BETA}.h5
done

# Summaries: mim_gauss
for BETA in 1.1 2.0 4.0 11.0
do
$py3 code/script/summarize_rfs.py \
    $DATA/runs/fig5/summ_mim_gauss_b${BETA}_ell.csv  `# Output Path` \
    $DATA/runs/fig5/pilot_rfs_mim_gauss_beta_$BETA.h5`# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv                `# Unit set` \
    $CODE/proc/rf_models/gaussian.py                 `# RF Model`
done
# Summaries: mim_flat
for BETA in 1.1 2.0 4.0 11.0
do
$py3 code/script/summarize_rfs.py \
    $DATA/runs/fig5/summ_mim_flat_b${BETA}_ell.csv   `# Output Path` \
    $DATA/runs/fig5/pilot_rfs_mim_flat_beta_$BETA.h5 `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv                `# Unit set` \
    $CODE/proc/rf_models/gaussian.py                 `# RF Model`
done
# Summaries: sens
for n in 100; do
for MODEL in sn1 sn4 sna; do echo $MODEL
for BETA in 1.1 2.0 4.0 11.0; do
$py3 code/script/summarize_rfs.py \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b${BETA}_ell.csv  `# Output ` \
    $DATA/runs/fig5/${MODEL}_rf_n${n}_b${BETA}.h5    `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv                `# Unit set` \
    $CODE/proc/rf_models/gaussian.py                 `# RF Model`
done; done; done

# -----------  Download  ----
for n in 100; do
for MODEL in sn1 sn4 sna; do echo $MODEL
for BETA in 1.1 2.0 4.0 11.0
do
    scp $sherlock:$SHRLK_DATA/runs/fig5/ell_${MODEL}_n${n}_b${BETA}_ell.csv $DATA/runs/fig5/ell_${MODEL}_n${n}_b${BETA}_ell.csv
done; done; done
# scp $sherlock:$SHRLK_DATA/runs/fig5/summ_mim_flat_b${BETA}_ell.csv $DATA/runs/fig5/summ_mim_flat_b${BETA}_ell.csv
# scp $sherlock:$SHRLK_DATA/runs/fig5/summ_mim_gauss_b${BETA}_ell.csv $DATA/runs/fig5/summ_mim_gauss_b${BETA}_ell.csv
# scp $sherlock:$SHRLK_DATA/runs/fig5/summ_sens_l1_b${BETA}_ell.csv $DATA/runs/fig5/summ_sens_l1_b${BETA}_ell.csv
# scp $sherlock:$SHRLK_DATA/runs/fig5/summ_sens_l3_b${BETA}_ell.csv $DATA/runs/fig5/summ_sens_l3_b${BETA}_ell.csv
# scp $sherlock:$SHRLK_DATA/runs/fig5/summ_sens_al_b${BETA}_ell.csv $DATA/runs/fig5/summ_sens_al_b${BETA}_ell.csv


# -------------------------------------------------------- Quiver maps ----

# Sizemap-quiver plots for each attention type
# for MODEL in 'mim_flat' 'mim_gauss' 'sens_l1' 'sens_l3' 'sens_al'; do echo $MODEL;
for n in 100; do
for MODEL in sn1 sn4 sna; do echo $MODEL;
for LAYER in '(0,1,0)' '(0,2,0)' '(0,3,0)' '(0,4,0)'; do echo $LAYER;
for BETA in 1.1 2.0 4.0 11.0
do
py3 code/script/plots/sizemap_quiver.py \
    $PLOTS/figures/fig5/quiver/quiv_${MODEL}_b${BETA}_l$LAYER.pdf `# Output` \
    $DATA/runs/270420/summ_base_ell.csv                   `# Base RFs ` \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b${BETA}_ell.csv   `# Cued RFs ` \
    $LAYER --lim 224 --em 2.1415                          `# Misc`
done; done; done; done

# -------------------------------------------------------- LinePlots ----

# Layer1 sensitivity
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig5/line_sens_l1.pdf              `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig5/summ_sens_l1_b1.1_ell.csv         `# Cued RFs` \
    $DATA/runs/fig5/summ_sens_l1_b2.0_ell.csv \
    $DATA/runs/fig5/summ_sens_l1_b4.0_ell.csv \
    $DATA/runs/fig5/summ_sens_l1_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --px_per_degree 22

# Mim-flat
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig5/line_mim_flat.pdf             `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig5/summ_mim_flat_b1.1_ell.csv        `# Cued RFs` \
    $DATA/runs/fig5/summ_mim_flat_b2.0_ell.csv \
    $DATA/runs/fig5/summ_mim_flat_b4.0_ell.csv \
    $DATA/runs/fig5/summ_mim_flat_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --px_per_degree 22

# Layer3 sensitivity
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig5/line_sens_l3.pdf              `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig5/summ_sens_l3_b1.1_ell.csv         `# Cued RFs` \
    $DATA/runs/fig5/summ_sens_l3_b2.0_ell.csv \
    $DATA/runs/fig5/summ_sens_l3_b4.0_ell.csv \
    --loc 56 56 --rad 1 \
    --disp "1.1" "2.0" "4.0" \
    --px_per_degree 22

# ------------------------------------------------- Size line plot

for n in 100; do
for MODEL in sn4; do echo $MODEL
$py3 code/script/plots/rf_size_dists.py \
    $PLOTS/figures/fig5/line_size_${MODEL}.pdf        `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b1.1_ell.csv        `# Cued RFs` \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b2.0_ell.csv  \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b4.0_ell.csv  \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b11.0_ell.csv \
    --loc 56 56 --rad 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv
done; done


# ------------------------------------------------- Shift line plot

for n in 100; do
for MODEL in sn4; do echo $MODEL
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig5/line_shift_${MODEL}.pdf      `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv              `# Base RFs` \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b1.1_ell.csv  `# Cued RFs` \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b2.0_ell.csv  \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b4.0_ell.csv  \
    $DATA/runs/fig5/ell_${MODEL}_n${n}_b11.0_ell.csv \
    --loc 56 56 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --px_per_degree 22 --rad 22 \
    --norm_summ $DATA/runs/270420/summ_base_ell.csv \
    --norm_param ">np.sqrt(n.major_sigma)" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv
done; done

# updated call for mim_gauss only : for manuscript
$py3 code/script/plots/rf_shift_dists.py \
    $PLOTS/figures/fig5/line_shift_mim_gauss.pdf      `# Output Path` \
    $DATA/runs/270420/summ_base_ell.csv               `# Base RFs` \
    $DATA/runs/fig5/summ_mim_gauss_b1.1_ell.csv       `# Cued RFs` \
    $DATA/runs/fig5/summ_mim_gauss_b2.0_ell.csv  \
    $DATA/runs/fig5/summ_mim_gauss_b4.0_ell.csv  \
    $DATA/runs/fig5/summ_mim_gauss_b11.0_ell.csv \
    --loc 56 56 --px_per_degree 22 --rad 22 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --figsize 5 5 --ylim -.22 .63 \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --norm_summ $DATA/runs/270420/summ_base_ell.csv \
    --norm_param ">np.sqrt(n.major_sigma)"

# ------------------------------------------------- Gain line plot

for n in 100; do
for MODEL in sn1; do echo $MODEL --no_$MODE
$py3 code/script/plots/act_dists.py \
    $PLOTS/figures/fig5/line_gain_${MODEL}_n${n}.pdf `# Output` \
    $DATA/runs/fig2/lenc_task_base.h5                 `# Base Acts` \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b1.1.h5       `# Cued RFs` \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b2.0.h5  \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b4.0.h5  \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b11.0.h5 \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --raw_ylim 0.01 1e6 \
    --sd_ylim .2 15 \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --figsize 5 5 --n_img 3 \
    --n_bins 100 --bootstrap_n 50
done; done; done

$py3 code/script/plots/act_dists.py \
    $PLOTS/figures/fig5/line_gain_mg_n${n}.pdf      `# Output Path` \
    $DATA/runs/fig2/lenc_task_base.h5               `# Base Acts` \
    $DATA/runs/fig5/lenc_mg_n100_b1.1.h5            `# Cued RFs` \
    $DATA/runs/fig5/lenc_mg_n100_b2.0.h5  \
    $DATA/runs/fig5/lenc_mg_n100_b4.0.h5  \
    $DATA/runs/fig5/lenc_mg_n100_b11.0.h5 \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --raw_ylim 0.2 1e6 --sd_ylim .2 15 \
    --figsize 5 5 --n_img 3 --no_read 

for n in 100; do
for MODEL in mg; do
for BETA in 1.1 2.0 4.0 11.0; do
scp $sherlock:$SHRLK_DATA/runs/fig5/lenc_${MODEL}_n${n}_b${BETA}.h5.sgain.npz $DATA/runs/fig5/lenc_${MODEL}_n${n}_b${BETA}.h5.sgain.npz
done; done; done

for MODE in "raw" "line"; do
for n in 100; do
for MODEL in sn1; do
scp $sherlock:/scratch/users/kaifox/attfield/plots/figures/fig5/line_gain_${MODEL}_n${n}_M${MODE}_zoom.pdf plots/figures/fig5/line_gain_${MODEL}_n${n}_M${MODE}_zoom.pdf
done; done; done

# zoom / detail view
for MODE in raw line; do
for n in 100; do
for MODEL in sn1; do echo $MODEL --no_$MODE
$py3 code/script/plots/act_dists.py \
    $PLOTS/figures/fig5/line_gain_${MODEL}_n${n}_M${MODE}_zoom.pdf `# Output` \
    $DATA/runs/fig2/lenc_task_base.h5                 `# Base Acts` \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b1.1.h5       `# Cued RFs` \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b2.0.h5  \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b4.0.h5  \
    $DATA/runs/fig5/lenc_${MODEL}_n${n}_b11.0.h5 \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --disp "1.1" "2.0" "4.0" "11.0" \
    --sd_ylim 0 6 \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --figsize 5 5 --n_img 3 \
    --n_bins 100 --bootstrap_n 50 --no_$MODE
done; done; done


cp $PLOTS/runs/280420/visual_flat_b4.0.pdf $PLOTS/figures/fig4/visual_flat_b4.0.pdf
```




<!-- ====================================================================== -->
<!-- ====================================================================== -->
## (c) “Gain matters” hypothesis in feature space
<!-- ====================================================================== -->
<!-- ====================================================================== -->


```bash
# ------------------ Download encodings.py output ----
# Gaussian mimicry encodings
scp $sherlock:$SHRLK_DATA/runs/fig5/enc_task_mim_gauss_b4.0.h5 $DATA/runs/fig5/enc_task_mim_gauss_b4.0.h5
# Flat mimicry encodings
scp $sherlock:$SHRLK_DATA/runs/fig5/enc_task_mim_flat_b4.0.h5 $DATA/runs/fig5/enc_task_mim_flat_b4.0.h5
# Layer1 sensitivity shift encodings
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_sens_l1_b4.0.h5 $DATA/runs/fig5/enc_task_sens_l1_b4.0.h5
# Layer3 sensitivity shift encodings
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_sens_l3_b4.0.h5 $DATA/runs/fig5/enc_task_sens_l3_b4.0.h5
# All-layer sensitivity shift encodings
scp $sherlock:$SHRLK_DATA/runs/fig4/enc_task_sens_al_b4.0.h5 $DATA/runs/fig5/enc_task_sens_al_b4.0.h5

for MODEL in 'mim_gauss' 'mim_flat' 'sens_l1' 'sens_l3' 'sens_al'
do echo $MODEL;
py3 code/script/plots/flex_readout.py \
    $PLOTS/figures/fig5/flex_{}_$MODEL.pdf       `# Out `\
    $DATA/models/logregs_iso224_t100.npz         `# Regs`\
    $DATA/runs/fig2/fnenc_task_base.h5           `# Base`\
    $DATA/runs/fig5/enc_task_${MODEL}_b4.0.h5    `# Cued`\
    '(0,4,3)' --jtr 0.15 --em 2.1415             `# Misc`
done

```



<!-- ====================================================================== -->
<!-- ====================================================================== -->
## (d) Don't match behavior of gaussian gain
<!-- ====================================================================== -->
<!-- ====================================================================== -->


```bash
# -----------  Download  ----
for BETA in 2.0 4.0
do
scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_sens_beta_${BETA}.h5 $DATA/runs/val_rst/bhv_sens_l1_beta_${BETA}.h5
scp $sherlock:$SHRLK_DATA/runs/val_rst/bhv_sens_l3_beta_${BETA}.h5 $DATA/runs/val_rst/bhv_sens_l3_beta_${BETA}.h5
scp $sherlock:$SHRLK_DATA/runs/fig5/bhv_sens_al_beta_${BETA}.h5 $DATA/runs/fig5/bhv_sens_al_beta_${BETA}.h5
done

# -----------  Beta-scaling: Shift Mimicry  ----
py3 $CODE/script/plots/bhv_by_beta.py \
    $PLOTS/figures/fig5/bhv_beta_shift.pdf          `# Output Path` \
    $DATA/runs/fig2/bhv_gauss_n600_beta_1.1.h5      `# Gaussian` \
    $DATA/runs/fig2/bhv_gauss_n600_beta_2.0.h5       \
    $DATA/runs/fig2/bhv_gauss_n600_beta_4.0.h5       \
    $DATA/runs/fig2/bhv_gauss_n600_beta_11.0.h5      \
    $DATA/runs/fig5/sn4_bhv_n300_b1.1.h5              `# Gaussian` \
    $DATA/runs/fig5/sn4_bhv_n300_b2.0.h5               \
    $DATA/runs/fig5/sn4_bhv_n300_b4.0.h5               \
    $DATA/runs/fig5/sn4_bhv_n300_b11.0.h5              \
    --disp "1.1" "2.0" "4.0" "11.0"                 `# Display names` \
           "1.1" "2.0" "4.0" "11.0"                  \
    --cond Gauss  Gauss  Gauss  Gauss                \
           Mim-Gauss  Mim-Gauss  Mim-Gauss  Mim-Gauss\
    --cmp $DATA/runs/val_rst/bhv_base.h5          `# Uncued control` \
    --cmp_disp Dist                               `# Control name` \
    --y_rng 0.5 1.0                               `# Standard scale` \
    --bar1 0.69 --bar2 0.87
for M in 2 3; do
for BETA in 1.1 2.0 4.0 11.0; do
scp $sherlock:$SHRLK_DATA/runs/fig5/sn${M}_bhv_n300_b${BETA}.h5 $DATA/runs/fig5/sn${M}_bhv_n300_b${BETA}.h5
done; done

# Sensitivity shift layer comparisons
py3 $CODE/script/plots/bhv_by_beta.py \
    $PLOTS/figures/fig5/bhv_beta_sens_norm_n300.pdf   `# Output Path` \
    $DATA/runs/fig5/bhv_sens_norm_l1_n300_b1.1.h5     `# Gaussian` \
    $DATA/runs/fig5/bhv_sens_norm_l1_n300_b2.0.h5      \
    $DATA/runs/fig5/bhv_sens_norm_l1_n300_b4.0.h5      \
    $DATA/runs/fig5/bhv_sens_norm_l1_n300_b11.0.h5     \
    $DATA/runs/fig5/sn4_bhv_n300_b1.1.h5              `# Gaussian` \
    $DATA/runs/fig5/sn4_bhv_n300_b2.0.h5               \
    $DATA/runs/fig5/sn4_bhv_n300_b4.0.h5               \
    $DATA/runs/fig5/sn4_bhv_n300_b11.0.h5              \
    $DATA/runs/fig5/bhv_sens_norm_al_n300_b1.1.h5     `# Gaussian` \
    $DATA/runs/fig5/bhv_sens_norm_al_n300_b2.0.h5      \
    $DATA/runs/fig5/bhv_sens_norm_al_n300_b4.0.h5      \
    $DATA/runs/fig5/bhv_sens_norm_al_n300_b11.0.h5     \
    --disp "1.1" "2.0" "4.0" "11.0"              `# Display names` \
           "1.1" "2.0" "4.0" "11.0"               \
           "1.1" "2.0" "4.0" "11.0"               \
    --cond L1 L1 L1 L1  \
           L4 L4 L4 L4  \
           AL AL AL AL  \
    --cmp $DATA/runs/fig2/bhv_base_n600.h5        `# Uncued control` \
    --cmp_disp Dist                               `# Control name` \
    --bar1 0.69 --bar2 0.87
scp $sherlock:$SHRLK_PLOTS/figures/fig5/bhv_beta_sens_norm_n300.pdf plots/figures/fig5/bhv_beta_sens_norm_n300.pdf
```



## TODO

__(4a)__ Mimicry diagram: not right, would like to talk trough -- figure out how to describe the fact we move receptive fields in layer 1 to effectively rewire layer 4.
Redo mim-gauss with more shift? Looking at encoding seems like basically no difference.

__(b)__ Line also? Stitched model shifts (torch bug workaround).

__(c)__ What units to select? Resize input images?

__(d)__ Only the human-matched strength?









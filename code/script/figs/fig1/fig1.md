

# Attfield Figure 1

> __Fig 1: We’re looking at human attention in this visual task__
> 
> (a) Task (for humans / computers)
> (b) Human behavior
> (c) Spatial gain in CNN, replicates human behavior (at high gain)

### (a) Task

Diagram reprenting dual human / computer task structure.

```bash
## Pull images from the task database, and generate a noise image
## these are later to be used in constructing the task diagram
code/script/figs/fig1/get_task_images.py
cp "data/imagenet_exemplars/Ferris wheel_6.png" \
    plots/figures/fig1/wheel_exemplar.png
```

### (b) Human behavior

Line / error bar of human performance in attended and unattended condition across duration.

```bash
## This one's quite simple, just produce the plot
code/script/figs/fig1/human_performance.py
```

### (c) CNN spatial gain

##### [1] Performance increases with attention gain.

```bash
py3 $CODE/script/plots/bhv_by_beta.py \
   $PLOTS/figures/fig1/gauss_beta_gain.pdf       `# Output Path` \
   $DATA/runs/val_rst/bhv_gauss_beta_1.1.h5      `# Gaussian` \
   $DATA/runs/val_rst/bhv_gauss_beta_2.0.h5       \
   $DATA/runs/val_rst/bhv_gauss_beta_4.0.h5       \
   $DATA/runs/val_rst/bhv_gauss_beta_11.0.h5      \
   --disp "1.1" "2.0" "4.0" "11.0"                   `# Display names` \
   --cmp $DATA/runs/val_rst/bhv_base.h5          `# Uncued control` \
   --cmp_disp Distributed  --y_rng 0.45 0.85 --jitter 0.06 \
   --metric acc  --figsize 4.5 3  --sns_context paper
```

##### [2] Maps onto human psychometric function.

```bash
## Some computation done in here of the psychometric functions
## but mostly just plotting
code/script/figs/fig1/human_performance.py
code/script/figs/fig1/psych_match.py
```


### Figure 2 : Network

```bash
py3 code/script/avg_rf.py \
   $DATA/runs/fig1/avg_rf_sizes_margin28.csv \
   $DATA/runs/270420/summ_base_ell.csv \
   28 224

```



## TODO

__(a)__ What's the minimum time? What does it say between exemplar images and trials? Truly 25-25? Highlight positive? How many trials?

__(b)__ Include $N$ __yep__? Switch to errorbar mean / SD to over subjects. Check w/ dan. Not square brackets. Correct choices (%) times 100.

__(c)__ [1]  Would love to have some measure of certainty / error, but separating points by category here while we merge over categories in the human plots doesn't make a lot of sense. Traces [2] Convert to grouping points over by subject. Pick new matching beta once we have real data. Vertical line connecting match points? Smaller duration range to spread information out? Pick particular point from human data. 




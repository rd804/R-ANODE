# Residual ANODE



For all cases `scripts/r_anode.py` is run with different arguments. The slurm submission script contains the arguments for `r_anode.py` used for each case. The submission loop script submits the slurm submission file, for different arguments $N_{sig}, w,$ split (retrainings of a different train-val split of the same dataset), seed (controls the seed for signal injection). 

Note: This project uses slurm job scheduler to submit parallel jobs.

## Background model


## Idealized version

Slurm submission: `r_anode_ideal.sh`

Submission loop: `r_anode_ideal_submit_loop.sh`

## Scanning fixed-w

Slurm submission: `r_anode_wscan.sh`

Submission loop: `r_anode_wscan_submit_loop.sh`

## Learnable w

Slurm submission: `r_anode_learnable.sh`

Submission loop: `r_anode_learnable_submit_loop.sh`

no-signal fit: `sbatch r_anode_no_signal_fit.sh`

## IAD and supervised

IAD: `sbatch BDT_IAD.sh`

Supervised: `sbatch BDT_supervised.sh`

## ANODE

## SIC curves (Figure 1 and 2)

## Samples (Figure 5)

## w-scan result (Figure 3)

`python scripts/w_scan.py`

## Learnable w results (Figure 4 and 6)


#!/bin/sh

for sig in 5 0.1 0.2 0.5 0.8 0.9 1 2 1.5 10
do 

    echo "Running for sigma = ${sig}"
    python scripts/gaussian_mixture_modified_em.py --sig_train ${sig} &> logs/gaussian_mixture_modified_with_weights_100_epochs_em_${sig}.log &

done
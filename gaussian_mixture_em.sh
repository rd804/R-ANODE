#!/bin/sh

for sig in 5 0.1 0.2 0.5 0.8 0.9 1 2 1.5 10
do 

    echo "Running for sigma = ${sig}"
    python scripts/gaussian_mixture_em.py --sig_train ${sig} &> logs/gaussian_mixture_em_${sig}.log &

done
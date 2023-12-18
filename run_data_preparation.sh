# wget https://zenodo.org/record/4536377/files/events_anomalydetection_v2.features.h5
# wget https://zenodo.org/record/5759087/files/events_anomalydetection_qcd_extra_inneronly_features.h5

# uncomment the above lines if the files are not already present

python scripts/run_data_preparation.py \
    --outdir data/lhc_co/ \

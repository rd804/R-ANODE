# wget https://zenodo.org/record/4536377/files/events_anomalydetection_v2.features.h5
# wget https://zenodo.org/record/5759087/files/events_anomalydetection_qcd_extra_inneronly_features.h5

python run_data_preparation_latentCATHODE_copy.py \
    --outdir input_data/ --S_over_sqrtB=-1

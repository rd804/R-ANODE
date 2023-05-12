# m-anode

Modification of ANODE: 

In the SB: Train a density estimator P_b(x)
In the SR: Train a density estimator P_data(x) = P_b(x) (interpolated) + epsilon * P_s (x)

Classifier = P_data (x) / P_b (x)


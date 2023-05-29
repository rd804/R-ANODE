# m-anode

Modification of ANODE: 


Classifier learns on training data: 1 + epsilon * P_s (x) / P_b (x)
Classifier on test data uses this likelihood ratio to classify
P_s(x)/P_b(x)

In the SB: Train a density estimator P_b(x)

In the SR: Train a density estimator P_data(x) = P_b(x) (interpolated) + epsilon * P_s (x)

Classifier = P_data (x) / P_b (x)




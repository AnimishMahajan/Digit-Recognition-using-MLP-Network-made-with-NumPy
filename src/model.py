import numpy as np
import os

import nn_npy as nn


dir = os.path.join("data", "processed", "mnist")

X = np.load((os.path.join(dir, "X.npy")))
Y = np.load((os.path.join(dir, "Y.npy")))

X_test = np.load((os.path.join(dir, "X_test.npy")))
Y_test = np.load((os.path.join(dir, "Y_test.npy")))

model1 = nn.network()
model1.model(
    Inputs = X,
    Outputs = Y,
    hidden_layers = [11],
    activation_func = ["relu", "softmax"],
    epochs = 10,
    learning_rate = 0.0001,
    batch = 10
)

model1.summary()

model1.train(X, Y, progress_bar=True)

print(f"\n\nTesting Accuracy : {model1.test_classification(X_test, Y_test)}%")

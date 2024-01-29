import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import functions

with open('train.p', 'rb') as f:
    X_train, y_train = pickle.load(f)
with open('test.p', 'rb') as f:
  X_test, y_test = pickle.load(f)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

mnist_model = [functions.layer_Dense(784, 128), 
        functions.activation_ReLu(), 
        functions.layer_Dense(128, 64),
        functions.activation_ReLu(),
        functions.layer_Dense(64, 10)]

optimizer = functions.Optimizer_Adam()

functions.train_model(mnist_model, 5, 1, 32, X_train, y_train, optimizer)

functions.test_model(mnist_model, X_test, y_test)



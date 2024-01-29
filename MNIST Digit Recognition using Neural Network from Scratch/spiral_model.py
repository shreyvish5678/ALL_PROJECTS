import numpy as np
import matplotlib.pyplot as plt
import functions
import spiral_data
from sklearn.model_selection import train_test_split

X_train, y_train = spiral_data.spiral_data(300, 3)
X_test, y_test = spiral_data.spiral_data(100, 3)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='brg')
plt.show()

spiral_model = [functions.layer_Dense(2, 64), 
        functions.activation_ReLu(), 
        functions.layer_Dense(64, 3)]

optimizer = functions.Optimizer_Adam()

functions.train_model(spiral_model, 101, 50, 4, X_train, y_train, optimizer)

functions.test_model(spiral_model, X_test, y_test)

labels = functions.predict(X_test, spiral_model)

print(labels.shape)

print(y_test.shape)

plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='brg', label='Actual')
plt.scatter(X_test[:,0], X_test[:,1], c=labels, cmap='brg', label='Predicted')
plt.show()

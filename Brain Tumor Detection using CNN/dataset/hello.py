from nnfslib import *
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
import matplotlib.pyplot as plt 
plt.imshow(x_train[1000], cmap='binary')
plt.show()
model = ([Flatten(), Dense(128), ReLu(), Dense(64), ReLu(), Dense(10)], 'Softmax')
#train_model(model, x_train, y_train, 10, 64, 'Adam')
eval_model(model, x_test, y_test)
save_model('model.p', model)
new_model = load_model('model.p')
predict_model(np.array([x_train[1000]]), new_model)

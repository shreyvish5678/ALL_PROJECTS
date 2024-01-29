import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

class layer_Dense:
  def __init__(self, input_size, neurons):
    self.weights = 0.10*np.random.randn(input_size, neurons)
    self.biases = np.zeros((1, neurons))
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases
  def backprop(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dinputs = np.dot(dvalues, self.weights.T) 
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
 
class activation_ReLu:
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)
  def backprop(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0 

class activation_Softmax: 
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss: 
  def calculate(self, output, y):
    sample_losses = self.forward(output, y)
    return np.mean(sample_losses)

class loss_catergorical_crossentropy(Loss):
  def forward(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    if len(y_true.shape) == 1: 
      correct_confidences = y_pred[range(samples), y_true] 
    elif len(y_true.shape) == 2: 
      correct_confidences = np.sum(y_pred*y_true, axis=1) 
    return -np.log(correct_confidences)

class activation_softmax_loss_catergorical_crossentropy():
  def __init__(self):
    self.activation = activation_Softmax()
    self.loss = loss_catergorical_crossentropy()

  def forward(self, inputs, y_true):
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)

  def backprop(self, dvalues, y_true):
    samples = len(dvalues)
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)
    self.dinputs = dvalues.copy()
    self.dinputs[range(samples), y_true] -= 1
    self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=0.01, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update_params(self, layer):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):              
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
          
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=0.001, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
        
class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

def train_model(model_arch, epochs, epoch_step, batch_size, train_features, train_labels, optimizer_given):
  softmax_loss = activation_softmax_loss_catergorical_crossentropy()
  optimizer = optimizer_given
  losses = []
  accuracies = []
  for epoch in range(epochs): 
    indices = np.arange(len(train_features)) 
    np.random.shuffle(indices) 
    train_features = train_features[indices] 
    train_labels = train_labels[indices]
      
    for i in range(0, len(train_features), batch_size):
      X_batch = train_features[i:i+batch_size] 
      y_batch = train_labels[i:i+batch_size] 

      (loss, accuracy) = forward_pass(model_arch, X_batch, y_batch, softmax_loss)

      backward_pass(model_arch, y_batch, softmax_loss, optimizer)

    if not epoch % epoch_step:      
      print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
    losses.append(loss)
    accuracies.append(accuracy)

  num_epochs = [epoch for epoch in range(epochs)]
  plt.plot(num_epochs, losses, 'b')
  plt.title('Loss over Epochs')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show()

  plt.plot(num_epochs, accuracies, 'b')
  plt.title('Accuracy over Epochs')
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.show()

def test_model(model_arch, test_features, test_labels):
  softmax_loss = activation_softmax_loss_catergorical_crossentropy()

  (test_loss, test_accuracy) = forward_pass(model_arch, test_features, test_labels, softmax_loss)

  print(f'Test Loss: {test_loss:.4f}') 
  print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

def forward_pass(model, X, y, softmax_loss_func):
  for num_layer in range(len(model)):
    if num_layer == 0:
      model[num_layer].forward(X)
    else:
      model[num_layer].forward(model[num_layer - 1].output)
  loss = softmax_loss_func.forward(model[len(model) - 1].output, y)
  predictions = np.argmax(softmax_loss_func.output, axis=1) 
  if len(y.shape) == 2: 
    y = np.argmax(y, axis=1) 
  accuracy = np.mean(predictions == y)
  return (loss, accuracy)

def backward_pass(model, y, softmax_loss_func, optimizer_func):
  softmax_loss_func.backprop(softmax_loss_func.output, y)
  for num_layer in range(len(model)):
    back_num = len(model) - 1 - num_layer
    if num_layer == 0:
      model[back_num].backprop(softmax_loss_func.dinputs)
    else:
      model[back_num].backprop(model[back_num + 1].dinputs)
  for layer in model:
    if isinstance(layer, layer_Dense):
      optimizer_func.update_params(layer)
      optimizer_func.post_update_params()

def predict(data, model):
  softmax = activation_Softmax()
  for num_layer in range(len(model)):
    if num_layer == 0:
      model[num_layer].forward(data)
    else:
      model[num_layer].forward(model[num_layer - 1].output)
  softmax.forward(model[len(model) - 1].output)
  return softmax.output
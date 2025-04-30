import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(sop):
    return 1.0 / (1 + np.exp(-1 * sop))

# Error function (Mean Squared Error)
def error(predicted, target):
    return np.power(predicted - target, 2)

# Derivative of the error function
def error_predicted_deriv(predicted, target):
    return 2 * (predicted - target)

# Derivative of the sigmoid function
def sigmoid_sop_deriv(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop))

# Derivative of the sum of products for weights (weights are scalars)
def sop_w_deriv(x):
    return x

# Update weights based on the gradients
def update_w(w, grad, learning_rate):
    return w - learning_rate * grad

# Initialize inputs and target
x1 = 0.1
x2 = 0.4
target = 0.7
learning_rate = 0.01

# Initialize weights randomly
w1 = np.random.rand()
w2 = np.random.rand()

print("Initial W: ", w1, w2)

# Lists to store predicted output and network error
predicted_output = []
network_error = []

# Initialize previous error (for comparison)
old_err = 0

# Training loop (for 80,000 iterations)
for k in range(80000):
    # Forward Pass
    y = w1 * x1 + w2 * x2
    predicted = sigmoid(y)
    err = error(predicted, target)
    predicted_output.append(predicted)
    network_error.append(err)
    
    # Backward Pass (Gradient Descent)
    g1 = error_predicted_deriv(predicted, target)
    g2 = sigmoid_sop_deriv(y)
    g3w1 = sop_w_deriv(x1)
    g3w2 = sop_w_deriv(x2)
    
    gradw1 = g3w1 * g2 * g1
    gradw2 = g3w2 * g2 * g1
    
    # Update weights using the gradients and learning rate
    w1 = update_w(w1, gradw1, learning_rate)
    w2 = update_w(w2, gradw2, learning_rate)

# Plot the network error over iterations
plt.figure()
plt.plot(network_error)
plt.title("Iteration Number vs Error")
plt.xlabel("Iteration Number")
plt.ylabel("Error")
plt.show()

# Plot the predicted output over iterations
plt.figure()
plt.plot(predicted_output)
plt.title("Iteration Number vs Prediction")
plt.xlabel("Iteration Number")
plt.ylabel("Prediction")
plt.show()

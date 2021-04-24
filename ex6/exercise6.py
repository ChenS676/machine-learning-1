import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset

# Load our two moons (I promise we will get a new dataset in the next exercise)
train_data = dict(np.load("two_moons.npz", allow_pickle=True))
test_data = dict(np.load("two_moons_test.npz", allow_pickle=True))
# we need to reshape our labels so that they are [N, 1] and not [N] anymore
train_samples, train_labels = train_data["samples"], train_data["labels"][:, None]
test_samples, test_labels = test_data["samples"], test_data["labels"][:, None]

def relu(x: np.ndarray) -> np.ndarray:
    """
    elementwise relu activation function
    :param x: input to function [shape: arbitrary]
    :return : relu(x) [shape: same as x]
    """
    return np.maximum(0, x)


def d_relu(x: np.ndarray) -> np.ndarray:
    """
    elementwise gradient of relu activation function
    :param x: input to function [shape: arbitrary]
    :return : d relu(x) / dx [shape: same as x]
    """
    ### TODO #########################
    ##################################
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    elementwise sigmoid activation function
    :param x: input to function [shape: arbitrary]
    :return : d sigmoid(x) /dx [shape: same as x]
    """
    ### TODO #########################
    ##################################
    return np.divide(1, 1+np.exp(-x))


def d_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    elementwise sigmoid activation function
    :param x: input to function [shape: arbitrary]
    :return : sigmoid(x) [shape: same as x]
    """
    ### TODO #########################
    ##################################
    return sigmoid(x)*(1-sigmoid(x))


def binary_cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    binary cross entropy loss (negative bernoulli ll)
    :param predictions: predictions by model (shape [N])
    :param labels: class labels corresponding to train samples, (shape: [N])
    :return binary cross entropy
    """
    ### TODO #########################
    ##################################
    N = len(predictions)
    bce = np.sum(labels * np.log(predictions), axis=0) + np.sum((1-labels)*np.log(1-predictions), axis=0)
    return - bce / N


def d_binary_cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    gradient of the binary cross entropy loss
    :param predictions: predictions by model (shape [N])
    :param labels: class labels corresponding to train samples, (shape [N])
    :return gradient of binary cross entropy, w.r.t. the predictions (shape [N])
    https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    """
    ### TODO #########################
    ##################################
    N = len(predictions)
    grad = np.divide(1 - labels, 1 - predictions) - np.divide(labels, predictions)
    return grad / N


def init_weights(neurons_per_hidden_layer: List[int], input_dim: int, output_dim: int, seed: int = 0) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    :param neurons_per_hidden_layer: list of numbers, indicating the number of neurons of each hidden layer
    :param input_dim: input dimension of the network
    :param output_dim: output dimension of the network
    :param seed: seed for random number generator
    :return list of weights and biases as specified by dimensions and hidden layer specification
    """
    # seed random number generator
    rng = np.random.RandomState(seed)
    scale_factor = 1.0
    prev_n = input_dim
    weights = []
    biases = []

    # hidden layers
    for n in neurons_per_hidden_layer:
        # initialize weights with gaussian noise
        weights.append(scale_factor * rng.normal(size=[prev_n, n]))
        # initialize bias with zeros
        biases.append(np.zeros([1, n]))
        prev_n = n

    # output layer
    weights.append(scale_factor * rng.normal(size=[prev_n, output_dim]))
    biases.append(np.zeros([1, output_dim]))

    return weights, biases


def forward_pass(x: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) \
        -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    propagate input through network
    :param x: input: (shape, [N x input_dim])
    :param weights: weight parameters of the layers
    :param biases: bias parameters of the layers
    :return: - Predictions of the network (shape, [N x out_put_dim])
             - hs: output of each layer (input + all hidden layers) (length: len(weights))
             - zs: preactivation of each layer (all hidden layers + output) (length: len(weights))
    """
    N = x.shape[0]
    hs = []  # list to store all inputs
    zs = []  # list to store all pre-activations
    # input to first hidden layer is just the input to the network
    h = x
    hs.append(h)
    b1, b2, b3 = biases
    W1, W2, W3 = weights
    ### TODO #########################
    # pass "h" to all hidden layers
    # record all inputs and pre-activations in the lists
    ########################################### layer 1 ############################################
    fc1 = x.dot(W1) + b1
    zs.append(fc1)

    y1 = relu(fc1)
    hs.append(y1)

    ########################################### layer 2 ############################################
    fc2 = y1.dot(W2) + b2
    zs.append(fc2)

    y2 = relu(fc2)
    hs.append(y2)

    ########################################### layer 3 ############################################
    fc3 = y2.dot(W3) + b3
    zs.append(fc3)

    ########################################### layer sigmoid ######################################
    y = sigmoid(fc3)
    hs.append(y)
    return y, hs, zs


def backward_pass(loss_grad: np.ndarray,
                  hs: List[np.ndarray], zs: List[np.ndarray],
                  weights: List[np.ndarray], biases: List[np.ndarray]) -> \
    Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    propagate gradient backwards through network
    :param loss_grad: gradient of the loss function w.r.t. the network output (shape: [N, 1])
    :param hs: values of all hidden layers during forward pass
    :param zs: values of all preactivations during forward pass
    :param weights: weight paramameters of the layers
    :param biases: bias parameters of the layers
    :return: d_weights: List of weight gradients - one entry with same shape for each entry of "weights"
             d_biases: List of bias gradients - one entry with same shape for each entry of "biases"
    """
    print("loss grad is ", loss_grad)
    X1, X2, X3, X4 = hs
    fc1, fc2, fc3 = zs
    W1, W2, W3 = weights
    b1, b2, b3 = biases
    N = loss_grad.shape[0]
    # return gradients as lists - we pre-initialize the lists as we iterate backwards
    d_weights = [None] * len(weights)
    d_biases = [None] * len(biases)

    ### TODO #########################
    d_X4 = loss_grad
    d_fc3 = d_sigmoid(fc3) * d_X4
    d_W3 = X3.T.dot(d_fc3)
    d_b3 = np.sum(d_fc3, axis=0)[None]
    d_weights[2] = d_W3
    d_biases[2] = d_b3


    d_X3 = d_fc3.dot(W3.T)
    d_fc2 = d_X3 * d_relu(fc2)
    d_W2 = X2.T.dot(d_fc2)
    d_b2 = np.sum(d_fc2, axis=0)[None, ...]
    d_weights[1] = d_W2
    d_biases[1] = d_b2

    d_X2 = d_fc2.dot(W2)
    d_fc1 = d_X2 * d_relu(fc1)
    d_W1 = X1.T.dot(d_fc1)
    d_b1 = np.sum(d_fc1, axis=0)[None, ...]
    d_weights[0] = d_W1
    d_biases[0] = d_b1
    return d_weights, d_biases


# hyper parameters
layers = [64, 64]
learning_rate = 1e-2

# init model
weights, biases = init_weights(layers, input_dim=2, output_dim=1, seed=42)

# book keeping
train_losses = []
test_losses = []

# Here we work with a simple gradient descent implementation, using the whole dataset at each iteration,
# You can modify it to stochastic gradient descent or a batch gradient descent procedure as an exercise
# for i in range(1000):
#
#     # predict network outputs and record intermediate quantities using the forward pass
#     prediction, hs, zs = forward_pass(train_samples, weights, biases)
#     train_losses.append(binary_cross_entropy(prediction, train_labels))
#
#     # compute gradients
#     loss_grad = d_binary_cross_entropy(prediction, train_labels)
#     w_grads, b_grads = backward_pass(loss_grad, hs, zs, weights, biases)
#
#     # apply gradients
#     for i in range(len(w_grads)):
#         print(w_grads[i])
#         print(b_grads[i])
#         weights[i] -= learning_rate * w_grads[i]
#         biases[i] -= learning_rate * b_grads[i]
#     test_losses.append(binary_cross_entropy(forward_pass(test_samples, weights, biases)[0], test_labels))
#
# # plotting
# plt.title("Loss")
# plt.semilogy(train_losses)
# plt.semilogy(test_losses)
# plt.legend(["Train Loss", "Test Loss"])


def plt_solution(samples, labels):
    plt_range = np.arange(-1.5, 2.5, 0.01)
    plt_grid = np.stack(np.meshgrid(plt_range, plt_range), axis=-1)
    plt_grid_shape = plt_grid.shape[:2]
    pred_grid = np.reshape(forward_pass(plt_grid, weights, biases)[0], plt_grid_shape)
    plt.contour(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=[0.5], colors=["black"])
    plt.contourf(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=10)
    plt.colorbar()
    s0 = plt.scatter(x=samples[labels[:, 0] == 0, 0], y=samples[labels[:, 0] == 0, 1],
                     label="c=0", c="blue")
    s1 = plt.scatter(x=samples[labels[:, 0] == 1, 0], y=samples[labels[:, 0] == 1, 1],
                     label="c=1", c="orange")
    plt.legend([s0, s1], ["c0", "c1"])
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.5, 1.5)


# plt.figure()
# plt.title("Trained Network - with train samples")
# plt_solution(train_samples, train_labels)
#
# plt.figure()
# plt.title("Trained Network - with test samples")
# plt_solution(test_samples, test_labels)
# plt.show()


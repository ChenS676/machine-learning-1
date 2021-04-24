import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.nn import Sequential
from torch.autograd import Variable
data_dict = dict(np.load("mnist.npz"))

# prepare data:
# - images are casted to float 32 (from uint8) mapped in interval (0,1) and a "fake" color channel is added.
#   torch uses "NCHW"-layout for 2d convolutions. (i.e., a batch of images is represented as a 4 d tensor
#   where the first axis (N) is the batch dimension, the second the (color) **C**hannels, followed by a **H**eight
#   and a **W**idth axis). As we have grayscale images there is only 1 color channel.
# - targets are mapped to one hot encoding - torch does that for us
with torch.no_grad():
    train_samples = torch.from_numpy(data_dict["train_samples"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    train_labels = torch.nn.functional.one_hot(torch.from_numpy(data_dict["train_labels"]))
    test_samples = torch.from_numpy(data_dict["test_samples"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    test_labels = torch.nn.functional.one_hot(torch.from_numpy(data_dict["test_labels"]))

# plot first 25 images in train setp
plt.figure(figsize=(25, 1))
for i in range(25):
    plt.subplot(1, 25, i + 1)
    # drop channel axis for plotting
    plt.imshow(train_samples[i, 0], cmap="gray", interpolation="none")
    plt.gca().axis("off")


classifier_fc = torch.nn.Sequential(
    torch.nn.Flatten(), # Flatten image into vector
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
    torch.nn.Softmax(dim=1)
    # Outputlayer: 10 neurons (one for each class), softmax activation
    ###########
)

classifier_conv = torch.nn.Sequential(
    # Conv Layer 1: 8 filters of 3x3 size, ReLU, Max Pool with size 2x2 and stride 2
    torch.nn.Conv2d(1, 8, 3),
    torch.nn.MaxPool2d(2, stride=2),
    torch.nn.ReLU(),

    # Conv Layer 2: 16 filters of 3x3 size, ReLU, Max Pool with size 2x2 and stride 2
    torch.nn.Conv2d(8, 16, 3),
    torch.nn.MaxPool2d(2, stride=2),
    torch.nn.ReLU(),
    # Flatten
    torch.nn.Flatten(),
    # Fully Connected Layer 1: 64 Neurons, ReLU
    torch.nn.Linear(16 * 5  * 5, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
    torch.nn.Softmax(dim=1)
    # Outputlayer: 10 neurons (one for each class), softmax activation
)


classifier = classifier_fc
#classifier = classifier_conv

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
#TODO

def cross_entropy_loss(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """ Cross entropy Loss:
    :param labels: Ground truth class labels (shape; [N, num_classes])
    :param predictions: predicted class labels (shape: [N, num_classes])
    :return: cross entropy (scalar)
    """
    return - torch.mean(torch.sum(labels * torch.log(predictions), axis=1))

batch_size = 64
from torch.utils.data import TensorDataset
train_loader = DataLoader(TensorDataset(train_samples, train_labels), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_samples, test_labels), batch_size=batch_size)
#####

epochs = 2  # small number of epochs should be sufficient to get descent performance

train_losses = []
test_losses = []


for i in range(epochs):
    print("Epoch {:03d}".format(i + 1))
    for batch in train_loader:
        img, labels = batch

        optimizer.zero_grad()
        #TODO##################
        # forward pass        #
        predictions = classifier(img)
        # backward pass       #
        loss = cross_entropy_loss(labels, predictions)
        # update step         #
        loss.backward()
        optimizer.step()
        #######################
        train_losses.append(loss.detach().numpy())

# Evaluate (we still need batching as evaluating all test points at once would probably melt your Memory)
avg_loss = avg_acc = 0
for batch in test_loader:
    samples, labels = batch
    predictions = classifier(samples)
    loss = cross_entropy_loss(labels, predictions)
    acc = torch.count_nonzero(predictions.argmax(dim=-1) == labels.argmax(dim=-1)) / samples.shape[0]

    avg_acc += acc / len(test_loader)
    avg_loss += loss / len(test_loader)


print("Test Set Accuracy: {:.3f}, Test Loss {:.3f}".format(avg_acc.detach().numpy(), avg_loss.detach().numpy()))

plt.figure()
plt.semilogy(train_losses)
plt.show()


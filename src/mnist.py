#!/usr/bin/env python3

## In this assignment you will learn how to train a model with early stopping with the help of the simple machine
## learning framework from the previous assignment. The intent is to show you how to train a machine learning model.
## It is very similar to what you would do in a real ML framework.
##
## We provide some code to download and load MNIST digits. The MNIST comes with separate test and training set, but not
## a validation set. Your task is to split the official "train" set to train and validation set and train the network
## with early stopping.
##
## Some notes:
##  * an EPOCH is a pass through the whole training set once.
##  * validation accuracy is the percentage of correctly classified digits on the validation set.
##  * early stopping is when you avoid overfitting the model by measuring validation accuracy every N steps (for example
##    every epoch) and stop your training when your model begins to get worse on the validation set. You can do that by
##    keeping track of the best validation accuracy so far as well as its epoch, and stopping if the validation
##    accuracy has not improved in the last M steps (e.g. last 10 epochs).
##    (A better way to do this is to keep the weights of the best performing model, but that is harder, since you need a
##    way to save and reload weights of the model. We keep it simple instead and use the last, slightly worse than the
##    best model).
##  * test accuracy is the percentage of correctly classified digits on the test set.
##  * watch out: if you load batch_size of data by numpy indexing, it is not guaranteed that you will actually get
##    batch_size of them: if your array length is not divisable by batch_size, you will get the remainder as the last
##    batch. Take that into account when calculating the percentages: use shape[0] to determine the real number of
##    elements in the current batch of data.
##  * verify() function is there in order to be used both in the validate() and test() functions for error measurement
##    without code duplication (just the input data should be different).
##  * this is a 10 way classification task, so your network will have 10 outputs, one for each digit. Such network
##    can be trained by SoftmaxCrossEntropyLoss (it is actually a Softmax layer followed by CrossEntropy, but it is
##    usually implemented in one function, because it is numerically more stable and easier to implement that way). So
##    for every image, you get 10 outputs. To figure which one is the correct class, you should find which is the most
##    active.
##  * MNIST comes with black-and-white binary images, with background of 0 and foreground of 255. Each image is 28x28
##    matrix. To feed that to the model, we flatten it to a 784 (28*28) length vector, and normalize it by 255, so the
##    background becomes 0 and the foreground 1.0. Labels are integers between 0-9. You don't have to worry about this,
##    it's already done for you. The networks usually like to receive an input in range -1 .. 1 or generally the mean
##    near 0, and the standard deviation near 1 (as the majority of MNIST pixels is black, normalizing it to 0..1 range
##    is good enough).
##  * You have to have the solution of the previous task in the same folder as this file, as it will load your previously
##    implemented functions from there.
##
## Your final test accuracy should be around 95% and should early stop after ~89 epochs.
##
## Scroll to the "# Nothing to do BEFORE this line." line, and let the fun begin! Good luck!

import os
from urllib import request
import gzip
import numpy as np

import framework as lib


class MNIST:
    FILES = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    URL = "http://yann.lecun.com/exdb/mnist/"

    @staticmethod
    def gzload(file, offset):
        with gzip.open(file, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)

    def __init__(self, set, cache="./cache"):
        os.makedirs(cache, exist_ok=True)

        for name in self.FILES:
            path = os.path.join(cache, name)
            if not os.path.isfile(path):
                print("Downloading " + name)
                request.urlretrieve(self.URL + name, path)

        if set == "test":
            f_offset = 2
        elif set == "train":
            f_offset = 0
        else:
            assert False, "Invalid set: " + set

        self.images = self.gzload(os.path.join(cache, self.FILES[f_offset]), 16).reshape(-1, 28 * 28).astype(
            np.float) / 255.0
        self.labels = self.gzload(os.path.join(cache, self.FILES[f_offset + 1]), 8)

    def __len__(self):
        return self.images.shape[0]


class SoftmaxCrossEntropyLoss:
    @staticmethod
    def _softmax(input):
        input = input - np.max(input, axis=-1, keepdims=True)
        e = np.exp(input)
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, net_output, targets):
        self.saved_variables = {
            "out": net_output,
            "targets": targets
        }

        out = self._softmax(net_output)
        return np.mean(-np.log(out[range(net_output.shape[0]), targets]))

    def backward(self):
        net_output = self.saved_variables["out"]
        targets = self.saved_variables["targets"]

        batch_size = net_output.shape[0]
        grad = self._softmax(net_output)
        grad[range(batch_size), targets] -= 1

        self.saved_variables = None
        return grad / batch_size


################################################################################

train_validation_set = MNIST("train")
test_set = MNIST("test")

n_train = int(0.7 * len(train_validation_set))
print("MNIST:")
print("   Train set size:", n_train)
print("   Validation set size:", len(train_validation_set) - n_train)
print("   Test set size", len(test_set))

np.random.seed(0xDEADBEEF)
batch_size = 64

loss = SoftmaxCrossEntropyLoss()
learning_rate = 0.03

model = lib.Sequential([
    lib.Linear(28 * 28, 20),
    lib.Tanh(),
    lib.Linear(20, 10)
])

#######################################################################################################################
# Nothing to do BEFORE this line.
#######################################################################################################################

indices = np.random.permutation(len(train_validation_set))


## Implement
## Hint: you should split indices to 2 parts: a training and a validation one. Later when loading a batch of data,
## just iterate over those indices by loading "batch_size" of them at once, and load data from the dataset by
## train_validation_set.images[your_indices[i: i+batch_size]] and
## train_validation_set.labels[your_indices[i: i+batch_size]]

train_indices = indices[:n_train]
validation_indices = indices[n_train:]
## End


def verify(images, targets):
    ## Implement
    ##  * verify() function is there in order to be used both in the validate() and test() functions for error measurement
    ##  without code duplication (just the input data should be different).
    y = model.forward(images)
    i = np.argmax(y, axis=1)

    num_ok = np.sum(i == targets)
    total_num = len(i)
    ## End

    return num_ok, total_num


def validate():
    accu = 0.0
    count = 0

    ## Implement. Use the verify() function to verify your data.
    for i in range(0, len(validation_indices), batch_size):
        images = train_validation_set.images[validation_indices[i: i+batch_size]]
        labels = train_validation_set.labels[validation_indices[i: i+batch_size]]

        ## Implement. Use the verify() function to verify your data.
        num_ok, total_num = verify(images, labels)

        accu += num_ok
        count += total_num
    ## End

    return accu / count * 100.0


def test():
    accu = 0.0
    count = 0

    for i in range(0, len(test_set), batch_size):
        images = test_set.images[i:i + batch_size]
        labels = test_set.labels[i:i + batch_size]

        ## Implement. Use the verify() function to verify your data.
        num_ok, total_num = verify(images, labels)

        accu += num_ok
        count += total_num
        ## End

    return accu / count * 100.0


## You should update these: best_validation_accuracy is the best validation set accuracy so far, best_epoch is the
## epoch of this best validation accuracy (the later can be initialized by anything, as the accuracy will for sure be
## better than 0, so it will be updated).
best_validation_accuracy = 0
best_epoch = -1

for epoch in range(1000):
    ## Implement
    mean_error = 0
    for i in range(0, len(train_indices), batch_size):
        images = train_validation_set.images[train_indices[i: i+batch_size]]
        labels = train_validation_set.labels[train_indices[i: i+batch_size]]

        error = lib.train_one_step(model, loss, learning_rate, images, labels)

        mean_error += (error * len(labels))

    error = mean_error / len(train_indices)
    validation_accuracy = validate()
    ## End

    print("Epoch %d: loss: %f, validation accuracy: %.2f%%" % (epoch, error, validation_accuracy))

    ## Implement
    ## Hint: you should check if current accuracy is better than the best so far. If it is not, check before how many
    ## iterations ago the best one came, and terminate if it is more than 10. Also update the best_* variables
    ## if needed.
    # end

print("Test set performance: %.2f%%" % test())

import numpy as np
import keras.backend as K



def get_weights(model):

    return model.get_weights()


def get_gradients(model):
    '''
    Return the gradient of every trainable weight in model
    '''
    weights = [tensor for tensor in model.trainable_weights]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from skimage.util import random_noise

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from copy import deepcopy
def create_permuted_mnist_task(num_datasets):
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    task_list = [mnist]
    for seed in range(1, num_datasets):
        task_list.append(permute(mnist, seed))
    return task_list

def permute(task, seed):
    np.random.seed(seed)
    perm = np.random.permutation(task.train._images.shape[1])
    permuted = deepcopy(task)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]
    return permuted


def create_disjoint_mnist_task():
    mnist = read_data_sets("MNIST_data/", one_hot=False)
    train_index = split_index(mnist.train.labels)
    test_index = split_index(mnist.test.labels)
    vali_index = split_index(mnist.validation.labels)
    mnist = read_data_sets("MNIST_data/", one_hot=True)

    train_set = {}
    test_set = {}
    vali_set = {}

    for i in range(3):
        train_set[i] = []
        test_set[i] = []
        vali_set[i] = []

        train_set[i].append(mnist.train.images[train_index[i]])
        train_set[i].append(mnist.train.labels[train_index[i]])

        test_set[i].append(mnist.test.images[test_index[i]])
        test_set[i].append(mnist.test.labels[test_index[i]])

        vali_set[i].append(mnist.validation.images[vali_index[i]])
        vali_set[i].append(mnist.validation.labels[vali_index[i]])

    return train_set,test_set,vali_set




def split_index(label):
    indi_1 = np.where(label<=3) 
    indi_2 = np.where(3<label<=6) 
    indi_3 = np.where(np.logical_and(6<label,label<=9))
    return [indi_1,indi_2,indi_3]





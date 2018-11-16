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


from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
from skimage.util import random_noise
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


def create_disjoint_cifar10_task():
    mnist = Cifar10()

    train_index = split_index(mnist.train.labels)
    test_index = split_index(mnist.test.labels)
    vali_index = split_index(mnist.validation.labels)
    return train_index, test_index, vali_index

    mnist.one_hot(True)

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
    indi_1 = np.where(label<=3)[0]
    indi_2 = np.where(np.logical_and(3<label,label<=6))[0]
    indi_3 = np.where(np.logical_and(6<label,label<=9))[0]
    return [indi_1,indi_2,indi_3]



class Data(object):

    def __init__(self):
        self.images = None
        self.labels = None


class Cifar10(object):

    def __init__(self,flag = False):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        self.train = Data()
        self.train.images = X_train[:45000]
        self.train.labels = y_train[:45000]
        self.validation = Data()
        self.validation.images = X_train[45000:]
        self.validation.labels = y_train[45000:]
        self.test = Data()
        self.test.images = X_test
        self.test.labels = y_test

        if flag is True:
            self.one_hot(flag)

    def one_hot(self,flag):
        if flag:

            self.train.labels = to_categorical(self.train.labels)
            self.validation.labels = to_categorical(self.validation.labels)
            self.test.labels = to_categorical(self.test.labels)

        else:
            self.train.labels = np.argmax(self.train.labels,axis=1)
            self.validation.labels = np.argmax(self.validation.labels,axis=1)
            self.test.labels = np.argmax(self.test.labels,axis=1)


























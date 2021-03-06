# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""
import numpy as np

def load_mnist(return_test = False, min_val = -1., max_val = 1., shuffle=True, seed = 1234567890):
    """
    Loads the MNIST Dataset
    """
    from keras.datasets import mnist
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape data to theano order (img_index, channels, height, width)
    x_train = x_train.reshape(x_train.shape[0],-1,x_train.shape[-2],x_train.shape[-1]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0],-1,x_test.shape[-2],x_test.shape[-1]).astype('float32')
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(y_train)
    
    if(return_test):
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train


def load_fashion_mnist(return_test = False, min_val = -1., max_val = 1., shuffle=True, seed = 1234567890):
    """
    Loads the Fashion-MNIST Dataset
    """
    
    from keras.datasets import fashion_mnist
    # Load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Reshape data to theano order (img_index, channels, height, width)
    x_train = x_train.reshape(x_train.shape[0],-1,x_train.shape[-2],x_train.shape[-1]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0],-1,x_test.shape[-2],x_test.shape[-1]).astype('float32')
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
     
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(y_train)
    
    if(return_test):
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train


def load_cifar10(return_test = False, min_val = -1., max_val = 1., shuffle=True, seed = 1234567890):    
    """
    Loads the Cifar10 Dataset
    """
    
    from keras.datasets import cifar10
        # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Reshape data to theano order (img_index, channels, height, width)
    x_train = x_train.reshape(x_train.shape[0],-1,x_train.shape[-2],x_train.shape[-1]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0],-1,x_test.shape[-2],x_test.shape[-1]).astype('float32')
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(y_train)
    
    if(return_test):
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train


def load_cifar100(return_test = False, min_val = -1., max_val = 1., shuffle=True, seed = 1234567890):
    """
    Loads the Cifar100 Dataset
    """
    from keras.datasets import cifar100
        # Load data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    # Reshape data to theano order (img_index, channels, height, width)
    x_train = x_train.reshape(x_train.shape[0],-1,x_train.shape[-2],x_train.shape[-1]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0],-1,x_test.shape[-2],x_test.shape[-1]).astype('float32')
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')   
    
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(y_train)
    
    if(return_test):
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train


def load_LSUN(return_test = False):
    """
    Loads the LSUN Dataset
    """
    raise NotImplementedError
    
    
def create_pairs(x, digit_indices, num_classes):
    """
    Based on the "create_pairs" function from:
        https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
        
    """
    pairs = []
    labels = []
    min_class_length = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for class_n in range(num_classes):
        for i in range(min_class_length):
            # Generate intra-class pair
            z1, z2 = digit_indices[class_n][i], digit_indices[class_n][i + 1]
            pairs += [[x[z1], x[z2]]]
            
            # Generate inter-class pair
            class_inc = np.random.randint(1, num_classes)
            dn = (class_n + class_inc) % num_classes
            z1, z2 = digit_indices[class_n][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            labels += [1, 0]
            
    return np.array(pairs), np.array(labels)


DATASET = {
	"mnist": load_mnist,
	"fashion_mnist": load_fashion_mnist,
	"cifar10": load_cifar10,
	"cifar100": load_cifar100,
    "LSUN" : load_LSUN,
    }
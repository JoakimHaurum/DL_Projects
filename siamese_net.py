# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum

This script borrows heavily from the Keras Mnist siamese example found at : https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
The code has been adapted to an object oriented structure, and I've tried to make it more adaptable for different datasets / easily adjustable architectures
"""

import numpy as np
from datasetLoader import DATASET, create_pairs
from losses import contrastive_loss
from layers import Distance_Layer
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, Flatten, Activation, MaxPool2D
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import time


class siamese_net(object):
    name = "SiameseNet"
    
    def __init__(self, epochs, batch_size, dataset, loss_path, result_path, checkpoint_path):
        creation_time = time.strftime('%Y%m%d-%H%M%S')        
        dir_prefix = self.name + "_" + creation_time + "/"
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.loss_path = dir_prefix+loss_path
        self.result_path = dir_prefix+result_path
        self.checkpoint_path = dir_prefix+checkpoint_path
        
        if self.dataset.lower() in DATASET:
            #Load dataset
            train_x, train_y, test_x, test_y = DATASET[self.dataset](return_test=True)
            num_classes = len(np.unique(train_y))
            
            # Create training and testing pairs
            train_digit_indices = [np.where(train_y == i)[0] for i in range(num_classes)]
            self.train_pair_x, self.train_pair_y = create_pairs(train_x, train_digit_indices, num_classes)

            test_digit_indices = [np.where(test_y == i)[0] for i in range(num_classes)]
            self.test_pair_x, self.test_pair_y = create_pairs(test_x, test_digit_indices, num_classes)
            
    
            print('image pair shape:', self.train_pair_x.shape)
            print('training pair count:', self.train_pair_y.shape[0])
            print('testing pair count:', self.test_pair_y.shape[0])
            
            #Set dimensions for input of discriminator
            self.input_shape = train_x.shape[1:]
            
        else:
            raise NotImplementedError
                    
        self.build()
        
        
    def build(self):
        self.siamese_net  = self.network(self.input_shape) 
        self.siamese_net.summary()
        
        self.siamese_net.compile(optimizer = Adam(), loss = contrastive_loss, metrics = [self.accuracy_metric])        
    
    
    def network(self, img_shape, base_feature_count = 128, scale_factor = 2):
        img_in = Input(img_shape)
        x = img_in
        
#        for s in range(scale_factor):
#            x = Conv2D(base_feature_count*2**s, (3, 3), strides=(1,1), padding='same', name = "Conv2D_"+str(s)+"a")(x)
#            x = BatchNormalization(name = "BN_"+str(s)+"a")(x)
#            x = Activation("relu", name = "ReLU_"+str(s)+"a")(x)
##            x = Conv2D(base_feature_count*2**s, (3, 3), strides=(1,1), padding='same', name = "Conv2D_"+str(s)+"b")(x)
##            x = BatchNormalization(name = "BN_"+str(s)+"b")(x)
##            x = Activation("relu", name = "ReLU_"+str(s)+"b")(x)
#            x = MaxPool2D(name = "MaxPool_"+str(s))(x)
#    
#        x = Flatten(name = "Flatten")(x)
#        x = Dense(128, name = "FC1")(x)
#        x = BatchNormalization(name = "BN_"+str(scale_factor+1))(x)
#        x = Activation("relu",name = "ReLU_"+str(scale_factor+1))(x)    
              
        
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        base_net = Model(img_in, x, name = "Base net")
        base_net.summary()
        
        input_a = Input(img_shape, name = "Input a")
        input_b = Input(img_shape, name = "Input b")
        
        out_a = base_net(input_a)
        out_b = base_net(input_b)
        
        distance = Distance_Layer([out_a,out_b])
        
        return Model([input_a, input_b], distance)
        
    
    def accuracy_metric(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    
    
    def evaluation_accuracy(self, y_true, y_pred, threshold = 0.5):
        pred = y_pred.ravel() < threshold
        return np.mean(pred == y_true)
    
    
    def fit(self):
        self.siamese_net.fit(x = [self.train_pair_x[:,0], self.train_pair_x[:,1]], y = self.train_pair_y, batch_size = self.batch_size, epochs = self.epochs)

    
    def predict(self, img_pair):
        return self.siamese_net.predict(img_pair)
    
    
    def pretty_print(self):
        print("\nepochs = \t\t{}\nbatch_size = \t\t{}\ndataset = \t\t{}\
              \nloss_path = \t\t{}\nresult_path = \t\t{}\ncheckpoint_path = \t{}\
              \ninput height = \t\t{}\ninput width = \t\t{}\ninput channels = \t{}".format(self.epochs,\
                      self.batch_size,self.dataset,\
                      self.loss_path,self.result_path,self.checkpoint_path,\
                      self.input_shape[1],self.input_shape[2],self.input_shape[0]))



if __name__ == "__main__":
    sNet = siamese_net(20, 128, "minst_fashion", "Loss_values", "Images", "Model_checkpoints")
    sNet.pretty_print()
    sNet.fit()
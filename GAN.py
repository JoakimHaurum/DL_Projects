# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

import numpy as np
from datasetLoader import DATASET
from utils import save_img_grid, save_loss_log, save_model
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Flatten, Activation, Reshape
from keras.optimizers import Adam
from keras.utils import plot_model
import os
import time


class GAN(object):
    name = "GAN"
    
    def __init__(self, epochs, batch_size, noise_dim, dataset, loss_path, result_path, checkpoint_path):
        creation_time = time.strftime('%Y%m%d-%H%M%S')        
        dir_prefix = self.name + "_" + creation_time + "/"
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dataset = dataset
        self.loss_path = dir_prefix+loss_path
        self.result_path = dir_prefix+result_path
        self.checkpoint_path = dir_prefix+checkpoint_path
        
        if self.dataset.lower() in DATASET:
            #Load dataset
            self.data_x, self.data_y = DATASET[self.dataset]()
            
            #Set dimensions for input of discriminator
            self.input_height = self.data_x.shape[-2]
            self.input_width = self.data_x.shape[-1]
            self.input_channels = self.data_x.shape[1]
            
            #Set dimensions for output of generator
            self.output_height = self.input_height
            self.output_width = self.input_width
            self.output_channels = self.input_channels
        
            #Number of sample generated images
            self.sample_count = 64
            
            #Number of mini batches
            self.batch_count = int(np.ceil(self.data_x.shape[0]/float(self.batch_size)))
            
        else:
            raise NotImplementedError
                    
        self.build((self.noise_dim,), ((self.input_channels, self.input_height, self.input_width)))
        
        
    def discriminator(self, img_shape, base_feature_count = 512, scale_steps = 4):
        assert base_feature_count / 2**(scale_steps-1) >= 1
        
        img_in = Input(img_shape, name = "D_Input")
        x = img_in
        
        x = Flatten(name = "Flatten")(x)
        
        for s in range(scale_steps):
            x = Dense(int(base_feature_count/2**s), name = "FC_"+str(s))(x)
            x = LeakyReLU(0.2, name = "LReLU_"+str(s))(x)
            #x = BatchNormalization(name = "BN_"+str(s))(x)
            
        x = Dense(1, name = "FC_"+str(scale_steps))(x)
        x = Activation("sigmoid", name = "Sigmoid")(x)
        
        return Model(img_in, x)
        
    
    def generator(self, noise_dim, img_shape, base_feature_count = 128, scale_steps = 3):
        noise_in = Input(noise_dim, name = "G_Input")
        x = noise_in
        
        for s in range(scale_steps):
            x = Dense(base_feature_count*2**s, name = "FC_"+str(s))(x)
            x = LeakyReLU(0.2, name = "LReLU_"+str(s))(x)
            x = BatchNormalization(name = "BN_"+str(s))(x)
        
        x = Dense(np.prod(img_shape), name = "FC_"+str(scale_steps))(x)
        x = Activation("tanh", name = "Tanh")(x)
        x = Reshape(img_shape, name = "Reshape")(x)
        
        return Model(noise_in, x)
    
    
    def build(self, z, img_shape):        
        self.disNet = self.discriminator(img_shape)
        self.disNet.compile(optimizer = Adam(lr=0.0002, beta_1 = 0.5), loss = "binary_crossentropy")
        
        self.genNet = self.generator(z, img_shape)
        noise_input = Input(shape=z)
        gen_out = self.genNet(noise_input)
        
        self.disNet.trainable = False
        dis_out = self.disNet(gen_out)
        
        self.gan = Model(noise_input, dis_out);
        self.gan.compile(optimizer = Adam(lr=0.0002, beta_1 = 0.5), loss = "binary_crossentropy")
        
       
        self.genNet.summary()
        self.disNet.summary()
        self.gan.summary()
#        plot_model(self.genNet, to_file='genNet.png')        
#        plot_model(self.disNet, to_file='disNet.png')        
#        plot_model(self.gan, to_file='GAN.png')        
    
    
    def fit(self, k = 1):
        
        if not os.path.exists(self.checkpoint_path): 
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.result_path): 
            os.makedirs(self.result_path)
        if not os.path.exists(self.loss_path): 
            os.makedirs(self.loss_path)
        
        training_history = {
                "generator" : [],
                "discriminator" : []}
        
        sample_generated_noise = np.random.uniform(-1.,1.,size = (self.sample_count,self.noise_dim)).astype(np.float32)
        
        save_img_grid(self.result_path, "/fake_imgs_0", self.generate(sample_generated_noise), 8, 8, "Untrained GAN")
        save_img_grid(self.result_path, "/real_imgs", self.data_x[np.random.randint(0,self.data_x.shape[0], size=self.sample_count)], 8, 8, "Real MNIST")
        
        
        for epoch in range(1,self.epochs+1):
            g_loss_epoch = []
            d_loss_epoch = []
            
            for idx in range(self.batch_count):  
                                
                #Train discriminator
                real_imgs = self.data_x[idx*self.batch_size:(idx+1)*self.batch_size]
                noise = np.random.uniform(-1.,1.,size = (real_imgs.shape[0],self.noise_dim)).astype(np.float32)
                fake_imgs = self.generate(noise)              
                
                fake_y  = np.zeros((real_imgs.shape[0], 1))
                real_y  = np.ones((real_imgs.shape[0], 1))
                
                d_loss_real = self.disNet.train_on_batch(real_imgs, real_y)
                d_loss_fake = self.disNet.train_on_batch(fake_imgs, fake_y)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                
                d_loss_epoch.append(d_loss)
                
                   
                #Train generator
                if idx % k == 0:
                    noise = np.random.uniform(-1.,1.,size = (self.batch_size,self.noise_dim)).astype(np.float32)
                    y = np.ones((self.batch_size, 1))
                    
                    g_loss = self.gan.train_on_batch(noise, y)
                    g_loss_epoch.append(g_loss)
            
            training_history["generator"].append(np.mean(np.array(g_loss_epoch), axis=0))
            training_history["discriminator"].append(np.mean(np.array(d_loss_epoch), axis=0))
            
            print("Epoch: [{}]\ng_loss: {:.4f}, d_loss: {:.4f}"
                      .format(epoch, training_history["generator"][-1], training_history["discriminator"][-1]))
            save_model(self.checkpoint_path, "/gen", self.genNet, epoch)
            save_model(self.checkpoint_path, "/dis", self.disNet, epoch)
            save_model(self.checkpoint_path, "/gan", self.gan, epoch)
            save_img_grid(self.result_path, "/fake_imgs"+"_"+str(epoch), self.generate(sample_generated_noise), 8, 8, "Epoch: " + str(epoch))
            save_loss_log(self.loss_path, "/loss"+"_"+str(epoch), training_history)
            
                
    
    def generate(self, noise_input):
        return self.genNet.predict(noise_input)
    
    
    def predict(self, img_input):
        return self.disNet.predict(img_input)
    
        
    def pretty_print(self):
        print("\nepochs = \t\t{}\nbatch_size = \t\t{}\nnoise_dim = \t\t{}\ndataset = \t\t{}\
              \nloss_path = \t\t{}\nresult_path = \t\t{}\ncheckpoint_path = \t{}\
              \ninput height = \t\t{}\ninput width = \t\t{}\ninput channels = \t{}\
              \noutput height = \t{}\noutput width = \t\t{}\noutput channels = \t{}\
              \ngen. sample count = \t{}\nbatch count = \t\t{}\n".format(self.epochs,\
                      self.batch_size,self.noise_dim,self.dataset,\
                      self.loss_path,self.result_path,self.checkpoint_path,\
                      self.input_height,self.input_width,self.input_channels,\
                      self.output_height, self.output_width, self.output_channels,\
                      self.sample_count, self.batch_count))

            
            

if __name__ == "__main__":
    gan = GAN(10, 64, 100, "mnist", "Loss_values", "Images", "Model_checkpoints")
    gan.pretty_print()
    gan.fit()
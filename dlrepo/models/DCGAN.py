# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, LeakyReLU, Flatten, Deconv2D, Activation, Reshape

from GAN import GAN

from utils.data_ops import determine_scale_factor

class DCGAN(GAN):
    """
    Class for constructing a Deep Convolutional Generative Adversarial Network
    
    Based on Radford et al. (2015) https://arxiv.org/abs/1511.06434
    """
    
    name = "DCGAN"
    
    def discriminator(self, img_shape, base_feature_count = 64):
        """
        Constrcuts a Convolutional discriminator for the DCGAN
        """     
        img_in = Input(img_shape, name = "D_Input")
        x = img_in
        
        min_value, scale_factor = determine_scale_factor(img_shape)
        
        for s in range(scale_factor):
            x = Conv2D(base_feature_count*2**s, (5,5), strides =(2,2), padding="same", name = "Conv_"+str(s))(x)
            x = BatchNormalization(name = "BN_"+str(s))(x)
            x = LeakyReLU(0.2, name = "LReLU_"+str(s))(x)
        
        x = Flatten(name = "Flatten")(x)
        x = Dense(100, name = "FC_"+ str(scale_factor))(x)
        x = BatchNormalization(name = "BN_"+str(scale_factor))(x)
        x = LeakyReLU(0.2,name = "LReLU_"+str(scale_factor))(x)
        x = Dense(1,name = "FC_"+str(scale_factor+1))(x)
        x = Activation("sigmoid", name = "Sigmoid")(x)
        
        return Model(img_in, x)
    

    def generator(self, noise_dim, img_shape, base_feature_count = 128):  
        """
        Constrcuts a Convolutional generatpr for the DCGAN
        """     
        noise_in = Input(noise_dim, name = "G_Input")
        x = noise_in
        
        min_value, scale_factor = determine_scale_factor(img_shape)

        x = Dense(min_value*min_value*base_feature_count, name = "FC_0")(x)
        x = BatchNormalization(name = "BN_0")(x)
        x = Activation("relu", name = "ReLU_0")(x)
        x = Reshape((base_feature_count,min_value,min_value), name = "Reshape")(x)
        
        for s in range(1,scale_factor):
            x = Deconv2D(int(base_feature_count/2**s), (5,5), strides = (2,2), padding="same", name = "DConv_"+str(s))(x)
            x = BatchNormalization(name = "BN_"+str(s))(x)
            x = Activation("relu", name = "ReLu_"+str(s))(x)
        
        x = Deconv2D(img_shape[0], (5,5), strides = (2,2), padding="same", name = "DConv_"+str(scale_factor))(x)
        x = Activation("tanh", name = "Tanh")(x)
        
        return Model(noise_in, x)
    
    
    
if __name__ == "__main__":
    gan = DCGAN(10, 64, 100, "mnist", "Loss", "Images", "Saved_models")
    gan.pretty_print()
    gan.fit()
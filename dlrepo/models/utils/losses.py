# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

from keras import backend as K

def contrastive_loss(y_true, y_pred):
    """
    Implementation of the contrastive loss function
    
    Static margin (TODO: set through function call)
    
    Based on Hadsell et al. (2006) https://cs.nyu.edu/~sumit/research/assets/cvpr06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1-y_true)*K.square(K.maximum(margin-y_pred,0)))
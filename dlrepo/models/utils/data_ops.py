# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

import numpy as np

def determine_scale_factor(img_shape, min_size = 4):
    """
    Determines how many times the input shape (CxHxW format, where H == W) can be halved, while the output shape is >= 4 and an integer dimension
    Returns both the 
    """
    c,h,w = img_shape
    
    assert type(c) == np.int
    assert type(h) == np.int
    assert type(w) == np.int
    
    scale_iter = 0
    value = h
    while(value%2 == 0 and value > min_size):
        scale_iter += 1
        value = value/2
        
    return int(value), int(scale_iter)


def scale(x, min_val = -1, max_val = 1, base_min = 0, base_max = 255):
    """
    Scales the input X from [base_min;base_max] to [min_val;max_val]
    """
    x = (max_val-min_val)*(x-base_min) / (base_max-base_min)+min_val
    print ("x min val: ", np.min(x))
    print ("x max val: ", np.max(x))
    
    return x


def standardize(x, mean = None, std = None):
    """
    Standardizes the input X so it has zero mean and unit variances across all channels for each feature/pixel 
    Either calculates the mean and std.dev or uses supplied mean and std.dev
    """
    
    if type(mean) == type(None) and type(std) == type(None):
        mean = x.mean(axis = 0)
        std = x.std(axis = 0)
        print ("x mean val: ", mean)
        print ("x std val: ", std)
    elif type(mean) != type(None) and type(std) != type(None):
        pass
    else:
        raise ValueError("The supplied mean and standard devaition has type {} and {}. You should supply either none or both of the metrics".format(type(mean), type(std)))
    x_std = (x - mean) / std
    print ("x min val: ", np.min(x_std))
    print ("x max val: ", np.max(x_std))
    
    return x_std, mean, std



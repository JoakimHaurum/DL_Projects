# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

def save_img_grid(dir_path, file_name, imgs, nh, nw, title = ""):
    assert imgs.shape[0] == nh*nw
    
    if imgs.shape[1] in [1, 3, 4]:
        imgs = imgs.transpose(0, 2, 3, 1)
    
    if imgs.shape[-1] == 1:
        imgs = np.squeeze(imgs)
        cmap = "gray"
    else:
        cmap = None
        
    plt.figure()
    plt.clf()
    plt.suptitle(title)
    
    for x in range(1, nh*nw+1):
        plt.subplot(nh, nw, x)
        plt.imshow(imgs[x-1], cmap = cmap)
        plt.axis('off')       
    plt.savefig('{}.pdf'.format(dir_path+file_name), bbox_inches="tight")
  
    
def save_model(dir_path, file_name, model, epoch):
    model.save(dir_path+file_name+"_"+str(epoch)+".hdf5")


def save_loss_log(dir_path, filename, loss_log):
    f = open(dir_path+filename,'wb')
    pickle.dump(loss_log, f, protocol = pickle.HIGHEST_PROTOCOL)
    f.close()


def determine_scale_factor(img_shape, min_size = 4):
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
    
    
    
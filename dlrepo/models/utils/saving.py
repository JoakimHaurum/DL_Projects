# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

def save_img_grid(dir_path, file_name, imgs, nh, nw, title = ""):
    """
    Saves the provided 2D images in a grid of size nw x nh in the designated directory    
    """
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
    """
    Saves the provided keras model .hdf5 file in the designated directory    
    """
    print(dir_path+file_name+"_"+str(epoch)+".hdf5")
    model.save(dir_path+file_name+"_"+str(epoch)+".hdf5")


def save_loss_log(dir_path, filename, loss_log):
    """
    Saves the provided loss log pickle file in the designated directory    
    """
    f = open(dir_path+filename,'wb')
    pickle.dump(loss_log, f, protocol = pickle.HIGHEST_PROTOCOL)
    f.close()
    
    
    
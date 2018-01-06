# -*- coding: utf-8 -*-
"""
@author: Joakim Bruslund Haurum
"""

from keras import backend as K
from keras.layers import Lambda


def Distance_Layer(inputs, function = "L2"):
    assert len(inputs) == 2
    
    if function in DIST_FUNCS:
        return Lambda(DIST_FUNCS[function], output_shape=output_shape, name = function+"_Dist")([inputs[0], inputs[1]])
    else:
        raise ValueError('The requested distance function "{}" is not supported'.format(function))


def L2_Distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def L1_Distance(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=1, keepdims=True)


def output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

DIST_FUNCS = {"L2" : L2_Distance,
        "L1" : L1_Distance}
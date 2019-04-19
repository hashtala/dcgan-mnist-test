# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:54:20 2019

@author: gela
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import Nafo.fixed_dcgan as gan
import matplotlib.pyplot as plt


dataframe = pd.read_csv('digits.csv')
X = np.array(dataframe.iloc[:, 2:])
del dataframe

X = np.float32(X/128)
X = X - 1
X = X.reshape(-1, 28, 28, 1)


model = gan.DCGAN(X, latent_dim = 100, activation = tf.nn.tanh)
model.fit(X, epoch = 320)

z = model.graph_and_get_image(100, False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

N = sess.run(z)
N = N + 1
generated_im = N[0,:,:,0]

plt.imshow(generated_im, cmap = 'Greys')


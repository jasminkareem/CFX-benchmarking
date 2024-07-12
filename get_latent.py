import os
import torch
import torchvision
import torch.nn as nn
import pickle
import pylab
import numpy as np
import scipy
import torch.optim as optim
import pandas as pd
#import torchvision.datasets as datasets
from tensorflow.keras.datasets import mnist 
import time
import tensorflow as tf
import alibi

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import euclidean
from scipy.stats import shapiro, normaltest

from torchvision import transforms
from torchvision.utils import save_image

from collections import Counter

from copy import deepcopy

from torch.autograd import Variable

from skimage.io import imread

# Local imports
from local_models import *
from helper_functions import *
from piece_hurdle_model import *
from optimize_explanations import *
from evaluation_metrics import *



"""
	Get Latent Input z for each test image using gradient descent
	For details see equation 1 of the paper by Kelly and Keane (2021).

	z =  arg min _(z0) ||C(G(z0)) - C(I)||(^2_2) + ||G(z0) - I||(^2_2)

	Args:
	- I: original input image x
	- G: GAN model
	- C: All layers except the last in trained neural network

	Returns: 
	- z_opt: Latent input optimized with gradient descent
	
"""



def objective_function(z0, I, C, G):
    # Difference between a generated instance classified using C and the actual classified instance 
    term1 = torch.norm(C(G(z0))[1] - C(I)[1])**2
    # Difference between generated instance and the actual instance
    term2 = torch.norm(G(z0) - I)**2
    return term1 + term2


def gradient_descent(I, C, G, z0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    z0.requires_grad = True
    optimizer = optim.SGD([z0], lr=learning_rate)
    
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = objective_function(z0, I, C, G)
        loss.backward()
        optimizer.step()
        
        #if torch.norm(z0.grad) < tol:
        #    break
    
    return z0.detach()



# Load models and data
G, C = load_models(MLP, Generator)
# classifierCNN = ClassifierCNN(cnn)
# croppedCNN = CroppedCNN(cnn)
train_loader, test_loader = load_dataloaders()
X_train, y_train, X_test, y_test = get_MNIST_data()
path = '/mnt/c/Users/Jasmin/Documents/PhDy1/nnv-xai-evaluation/AAAI-2021-semifactual/AAAI-2021-master/data/latent_z_mnist_9_200'

'''
for sample_num in range(1): #On entire set: len(test_loader)
    original_query_idx, original_query_img, original_query_label = get_classification(test_loader, C, sample_num)
    batch_size = original_query_img.size(0) 
    # Sample z0 from a standard normal distribution
    #z0 = torch.randn_like(original_query_img)
    device = torch.device("cpu")
    z0 = torch.randn(1, 100, 1, 1, device=device)

    # Find optimal z
    z_opt = gradient_descent(I=original_query_img, C=C, G=G, z0=z0)

    # save optimal z
    filename = "z_opt_MNISTsample_" + str(sample_num) + ".pt"
    save_path = os.path.join(path, filename)
    torch.save(z_opt, save_path)

'''

# check test accuracy of neural network
accurate_instances =  0

for i, data in enumerate(test_loader):
    # This is too slow: original_query_idx, original_query_img, original_query_label = get_classification(test_loader, C, sample_num)
    image, label = data
    predicted_class = torch.argmax(C(image)[0]).detach().numpy()
    label = label.detach().numpy()[0]
    if predicted_class == label:
        accurate_instances += 1


print("test accuracy:", accurate_instances/len(test_loader))
# test accuracy of mnist_9_200: 0.5752 







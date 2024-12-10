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
from local_models_piece import *
from helper_functions_piece import *
from piece_hurdle_model import *
from optimize_explanations_piece import *
from evaluation_metrics import *



"""
	Get Latent Input z for each test image and target class
	For details see equation 1 of the paper by Kelly and Keane (2021).

	z =  arg min _(z0) ||C(G(z0)) - C(I)||(^2_2) + ||G(z0) - I||(^2_2)

	Args:
	- I: original input image x
	- G: GAN model
	- C: All layers except the last in trained neural network

	Returns: 
	- z: Latent input z to be optimized
	
"""



def objective_function(z0, I, C, G):
    # Difference between a generated instance classified using C and the actual classified instance
    #print(C(G(z0))[1]) 
    #print(C(I)[1])
    term1 = torch.norm(C(G(z0))[0] - C(I)[0])**2
    # Difference between generated instance and the actual instance
    term2 = torch.norm(G(z0) - I)**2
    return term1 + term2



# Load models and data
print('-- mnist_schut_mlp --')
path_classifier = '/mnt/c/Users/Jasmin/Documents/PhDy1/nnv-xai-evaluation/AAAI-2021-semifactual/AAAI-2021-master/weights/mnist_schut_mlp.pt'
#path_classifier = 'weights/mnist_9_200_nat.pth'
G, C = load_models(CNN, Generator, path_classifier, "mnist_schut_mlp")
# classifierCNN = ClassifierCNN(cnn)
# croppedCNN = CroppedCNN(cnn)
train_loader, test_loader = load_dataloaders()
X_train, y_train, X_test, y_test = get_MNIST_data()
#path = '/mnt/c/Users/Jasmin/Documents/PhDy1/nnv-xai-evaluation/AAAI-2021-semifactual/AAAI-2021-master/data/latent_z_mnist_9_200'
path = '/mnt/c/Users/Jasmin/Documents/PhDy1/nnv-xai-evaluation/AAAI-2021-semifactual/AAAI-2021-master/data/latent_z_mnist_schut_mlp'
lr = 0.01
epochs = 3000


for sample_num in range(100): #On entire set: len(test_loader)
    original_query_idx, original_query_img, original_query_label = get_classification(test_loader, C, sample_num)
    original_query_pred = int(torch.argmax(C(original_query_img)[0]).detach().numpy())
    print("instance:", sample_num)
    print("correct label:", original_query_label)
    print("prediction:", original_query_pred)



    batch_size = original_query_img.size(0) 

    # Sample z0 from a standard normal distribution
    device = torch.device("cpu")
    z = torch.randn(1, 100, 1, 1, device=device)

    # define optimizer and loss
    z.requires_grad = True
    optimizer = optim.Adam([z], lr=lr)

    for epoch in range(epochs):
        loss = objective_function(z0=z, I=original_query_img, C=C, G=G)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # save optimal z
    filename = "z_opt_MNISTsample_" + str(sample_num) + ".pt"
    save_path = os.path.join(path, filename)
    torch.save(z, save_path)




####### check test accuracy of neural network #######
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
# test accuracy of mnist_cnn_6_128: 







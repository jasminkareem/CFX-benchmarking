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

'''
	Equation 2:

	arg max||S(C(G(z))) - Yc|| 
	   z 
	   Adapting to find the 3 closest decision boundaries
'''


def find_cf_gradient_ascent(z, lr, G, original_query_label, cnn, pred_val=None):

	z_temp = z.clone().detach().requires_grad_(True)
	optimizer = optim.Adam([z_temp], lr=lr)
	criterion = torch.nn.MSELoss()
	
	
	output_t = torch.tensor([0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
	output_t[original_query_label] = 1
	epoch = 0

	while epoch < 10000:
		epoch += 1
		optimizer.zero_grad()

		output_e_raw = cnn(G(z_temp))[0] 
		output_e = torch.nn.functional.softmax(output_e_raw, dim=1)[0]
		loss = criterion(output_e, output_t)
		(-loss).backward(retain_graph=True)  
		optimizer.step()

		pred = int(torch.argmax(output_e))
		#print('pred:', pred)
		if pred_val == None:
			if pred != original_query_label:
				return pred
			else:
				continue
		else: 
			if pred != original_query_label and pred not in pred_val:
				return pred
			else:
				continue

	return None

			

		

path_classifier = 'weights/mnist_relu_4_1024.pt'
latent_file_path = 'data/latent_z_mnist_relu_4_1024/z_opt_MNISTsample_'
G, C = load_models(Mnist_relu_4_1024, Generator, path_classifier)
lr = 0.01
train_loader, test_loader = load_dataloaders()

preds = {}

for sample_num in range(100):
	temp_preds = []
	z = torch.load(latent_file_path + str(sample_num) + ".pt")
	original_query_idx, original_query_img, original_query_label = get_classification(test_loader, C, sample_num)
	if int(torch.argmax(C(original_query_img)[0]).detach().numpy()) != original_query_label:
		temp_preds.append(original_query_label)

	pred1 = find_cf_gradient_ascent(z, lr, G, original_query_label, C, pred_val=None)
	temp_preds.append(pred1)
	print('----> done pred1:', pred1)
	if pred1 == None:
		preds[str(sample_num)] = temp_preds
		continue
		
	pred2 = find_cf_gradient_ascent(z, lr, G, original_query_label, C, pred_val=temp_preds)
	temp_preds.append(pred2)
	print('----> done pred2:', pred2)
	if len(temp_preds) == 3:
		pass
	else:
		pred3 = find_cf_gradient_ascent(z, lr, G, original_query_label, C, pred_val=temp_preds)
		temp_preds.append(pred3)

	print('-----> done pred3:', temp_preds[-1])

	preds[str(sample_num)] = temp_preds

pd.DataFrame.from_dict(data=preds, orient='index').to_csv('target_classes_100.csv', header=False)
	



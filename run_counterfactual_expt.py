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


# Load models and data
G, cnn = load_models(MLP, Generator)
# classifierCNN = ClassifierCNN(cnn)
# croppedCNN = CroppedCNN(cnn)
train_loader, test_loader = load_dataloaders()
X_train, y_train, X_test, y_test = get_MNIST_data()

##### If working with new model, run the code below to save pickle #####
new_model = True
if new_model == True: 
	#collected_data is a dictionary where each key represents a class, and the corresponding value is a list containing activations for that class.
	collected_data = return_feature_contribution_data(train_loader, cnn)
	pred_features = {}

	#Assuming num_classes is the number of classes
	num_classes = 10

	#Create an empty list for each class
	for class_name in range(num_classes):
		pred_features[class_name] = {'activations': []}

	#Fill data from pred_idx_train into dist_data
	for class_name, activations_list in collected_data.items():
			#Convert activations_list to a numpy array
		activations_array = np.array(activations_list)
			#Store activations_array into the corresponding class in dist_data
		pred_features[class_name]['activations'] = activations_array

	# save resulting dictionary into a pickle
	model_name = 'MNIST_9_200'
	#with open('pred_features_'+ model_name + '.pickle', 'wb') as handle:
	#	pickle.dump(pred_features, handle, protocol=pickle.HIGHEST_PROTOCOL)




# define which pickle file to use for chosen trained neural network
picklefile = 'pred_features_MNIST_9_200.pickle'
#picklefile = 'pred_features.pickle'


expt1_data = pd.DataFrame(columns=['optim_time', 'IM1', 'IM2', 'Instance', 'Name', 'MC-Mean', 'MC-STD', 'NN-Dist'])


# k-NN for NN-Dist
X_train_act = np.load("data/distribution_data/X_train_act.npy")
X_test_act = np.load("data/distribution_data/X_test_act.npy")
X_train_pred = np.load("data/distribution_data/X_train_pred.npy")
X_test_pred = np.load("data/distribution_data/X_test_pred.npy")
k_nn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
k_nn.fit(X_train_act, X_train_pred)

# Loading AEs for IM1 and IM2 metrics
aes, ae_full = load_autoencoders()

# Probabilitiy threshold for identifying "Exceptional Features" with PIECE
alpha = 0.05

quit()

# Iterate though samples from MNIST
for sample_num in range(3):
	for target_class in range(10):
		# Get Query representations
		original_query_idx, original_query_img, original_query_label = get_classification(test_loader, cnn, sample_num)
		
		#original_query_idx, original_query_img, target_class = get_missclassificaiton(test_loader, cnn, rand_num)
		original_query_pred = int(torch.argmax(cnn(original_query_img)[0]))

		if target_class == original_query_pred: 
			# skip target classes where the target is the same as the predicted class of the instance
			continue

		print("Target class x':", target_class)
		# load latent dataset/features L for test image i?
		z = torch.load("data/latent_z_mnist_9_200/z_opt_MNISTsample_" + str(sample_num) + ".pt")
		#z = torch.load("data/latent_g_input_saved/mnist9_200_latent/sample_" + str(sample_num) + "_label_" + str(original_query_label) + "_pred_" + str(original_query_pred) + "_target_"  + str(target_class) ".pt")
		query_activations = cnn(G(z))[1][0]



		#### ========== First two steps of PIECE Algorithm ========== ####
		# Step 1: Acquire the probability of each features, and identify the excpetional ones (i.e., those with a probability lower than alpha)
		df = acquire_feature_probabilities(target_class, cnn, original_query_img=original_query_img, alpha=alpha, picklefile=picklefile) 
		# Step 2: Filter out exceptional features which we want to change, and change them to their expected values in the counterfactual class
		df = filter_df_of_exceptional_noise(df, target_class, cnn, alpha=alpha)
		# Sort by least probable to the most probable
		df = df.sort_values('Probability of Event')
		# Get x' -- The Ideal Explanation
		ideal_xp = modifying_exceptional_features(df, target_class, query_activations)   
		ideal_xp = ideal_xp.clone().detach().float().requires_grad_(False)



		for name in ['PIECE', 'Min-Edit', 'C-Min-Edit']:  # 'CEM', 'Proto-CF']:

			print(" ")
			print("-------------------------------")
			print(sample_num, name)
			print("-------------------------------")

			cnn = cnn.eval()
			temp_data = pd.DataFrame()

			# Query
			x_q = cnn(G(z))[1][0]

			# Explanation latent input (to optimize...)
			z_e = z.clone().detach().float().requires_grad_()

			criterion = nn.MSELoss()

			start_time = time.time()

			if name == 'PIECE':
				optimizer = optim.Adam([z_e], lr=0.01)
				z_e = optim_PIECE(G, cnn, ideal_xp, z_e, criterion, optimizer)

			elif name == 'Min-Edit':
				optimizer = optim.Adam([z_e], lr=0.001)
				z_e = optim_min_edit(cnn, G, z_e, optimizer, target_class)

			elif name == 'C-Min-Edit':
				optimizer = optim.Adam([z_e], lr=0.001)
				# z_e = optim_c_min_edit(G, cnn, x_q, z_e, criterion, optimizer, target_class)

			elif name == 'CEM':
				xp = optim_CEM_Explanation(original_query_idx)
				try:
					if xp == None:
						print("Couldn't Find Explanation")
						continue
				except:
					print('Found Explanation')

			elif name == 'Proto-CF':
				xp = optim_Proto_Explanation(original_query_idx)
				try:
					if xp == None:
						print("Couldn't Find Explanation")
						continue
				except:
					print('Found Explanation')


			optim_time = time.time() - start_time
			# Get explanation I_e
			if name == 'PIECE' or name == 'Min-Edit' or name == 'C-Min-Edit':
				I_e = G(z_e)
				
			elif name == 'CEM' or name == 'Proto-CF':
				I_e = torch.tensor(xp, dtype=torch.float32).reshape(-1,1,28,28)

			save_name = name
			#Save as pdf
			save_query_and_gan_xp_for_final_data(I_e, cnn, z, G, z_e, original_query_img, save_name, sample_num)
			
			# New prediction of explanation x'
			new_pred = int(torch.argmax(torch.exp(  cnn(I_e)[0]  )))
			
			# Metrics for Plausibility
			mc_dropout_results = mc_dropout(cnn, new_pred, I_e)
			nn_dist, _ = k_nn.kneighbors(X=np.array(    cnn(I_e)[1].detach().numpy()  )  , n_neighbors=2)
			IM1 = IM1_metric(I_e, aes, original_query_pred, new_pred)
			IM2 = IM2_metric(I_e, aes, ae_full, new_pred)

			temp_data = temp_data.append({'Instance': sample_num, 'Name': name, 'MC-Mean': mc_dropout_results.mean(),
									'MC-STD': mc_dropout_results.std(), 'NN-Dist': nn_dist[0][0], 'IM1': IM1,
									'IM2': IM2, 'optim_time': optim_time}, ignore_index=True) 


			print(temp_data.head())
			expt1_data = pd.concat([expt1_data, temp_data])
	
	print(expt1_data.head())
	expt1_data.to_csv('output_MNIST_data.csv', index=False)
	print("Time to do one digit:", round(time.time() - start_time, 3))


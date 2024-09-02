import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim
import pandas as pd
import time
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from scipy.stats import shapiro, normaltest
import argparse


# Local imports
from local_models import *
from helper_functions import *
from piece_hurdle_model import *
from optimize_explanations import *
from evaluation_metrics import *

def main():
	# Add arguments
	parser = argparse.ArgumentParser(description="Experiments for PIECE, Min-edit and C-Min-Edit")
	parser.add_argument('--path_classifier', help='Path to the image classifier file', default='weights/mnist_relu_4_1024.pt')
	parser.add_argument('--model_name', help='Name of image classifier', default=Mnist_relu_4_1024)
	parser.add_argument('--generator_name', help='Name of generator', default=Generator)
	parser.add_argument('--path_generator', help='Path to the generator file', default='weights/generator.pth')
	parser.add_argument('--dataset', help='dataset that you would like to use. Default: MNIST', default='mnist')
	parser.add_argument('--new_model', help='Whether or not you are using a new model and hence need to save the pickle file containing activations', default=False)
	parser.add_argument('--pickle_file_name', help='Name of pickle file for given trained neural network', default='relu_4_1024')
	parser.add_argument('--pickle_file', help='Path to pickle file', default='pred_features_relu_4_1024.pickle')
	parser.add_argument('--n_samples', help='Number of samples/instances you want to create explanations for', default=10) 
	parser.add_argument('--path_explanations', help='Path to output explanations and original instance', default='/mnt/c/Users/Jasmin/Documents/PhDy1/nnv-xai-evaluation/mnist/')
	parser.add_argument('--latent_file_path', help='Location of latent files', default="data/latent_z_mnist_relu_4_1024/z_opt_MNISTsample_")
	parser.add_argument('--target_classes', help='File with target classes for each instances', default='target_classes_1.csv')

	args = parser.parse_args()


	# Define arguments into variables
	path_classifier = args.path_classifier #'weights/mnist_relu_4_1024.pt'
	model = args.model_name #MLP
	Gen = args.generator_name #Generator
	path_generator = args.path_generator
	dataset = args.dataset # mnist
	new_model = args.new_model #True or false
	model_name = args.pickle_file_name #pickle file name - pred_features
	picklefile = args.pickle_file
	n_samples = args.n_samples 
	path_explanations = args.path_explanations
	latent_file_path = args.latent_file_path
	target_classes = args.target_classes




	# Start
	G, cnn = load_models(MLP=model, Generator=Gen, path_classifier=path_classifier, generator_file=path_generator)

	if dataset == 'mnist': 
		train_loader, test_loader = load_dataloaders()
		X_train, y_train, X_test, y_test = get_MNIST_data()

	else: 
		print("add dataloaders here")
		quit()

	##### If working with new model, run the code below to save pickle #####
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
		with open('pred_features_'+ model_name + '.pickle', 'wb') as handle:
			pickle.dump(pred_features, handle, protocol=pickle.HIGHEST_PROTOCOL)




	expt1_data = pd.DataFrame(columns=['Instance', 'Name', 'optim_time','Original Label', 'Original Prediction', 'Target Label', 'New Prediction'])


	# k-NN for NN-Dist

	#k_nn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
	#k_nn.fit(X_train_act, X_train_pred)

	# Loading AEs for IM1 and IM2 metrics
	#aes, ae_full = load_autoencoders()

	# Probabilitiy threshold for identifying "Exceptional Features" with PIECE
	alpha = 0.05

	#targets =  pd.read_csv(target_classes)
	#print(targets.head())
	target_classes = [1,3,5,8,9]



	# Iterate though samples 
	for sample_num in range(n_samples):
		# Get Query representations
		original_query_idx, original_query_img, original_query_label = get_classification(test_loader, cnn, sample_num)
		#original_query_idx, original_query_img, target_class = get_missclassificaiton(test_loader, cnn, rand_num)
		original_query_pred = int(torch.argmax(cnn(original_query_img)[0]).detach().numpy())

		# save original image
		with open(path_explanations + 'original/instance_' + str(sample_num) + '.txt', 'w') as outfile:
			np.savetxt(outfile, original_query_img.detach().numpy()[0][0])


		# Add code to find target class here 
		


		print('sample:', sample_num)
		for target_class in target_classes:

			if target_class == original_query_pred: 
				# skip target classes where the target is the same as the predicted class of the instance
				continue

			print("Correct label x:", original_query_label)
			print("Original prediction f(x):", original_query_pred)
			print("Target class x':", target_class)
			
			# load latent dataset/features L for test image i?
			z = torch.load(latent_file_path + str(sample_num) + ".pt")
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

				# Explanation latent input 
				z_e = z.clone().detach().float().requires_grad_()

				criterion = nn.MSELoss()

				start_time = time.time()

				if name == 'PIECE':
					optimizer = optim.Adam([z_e], lr=0.001)
					z_e = optim_PIECE(G, cnn, ideal_xp, z_e, criterion, optimizer, target_class)

				elif name == 'Min-Edit':
					optimizer = optim.Adam([z_e], lr=0.001)
					z_e = optim_min_edit(cnn, G, z_e, optimizer, target_class)

				elif name == 'C-Min-Edit':
					optimizer = optim.Adam([z_e], lr=0.001)
					z_e = optim_c_min_edit(G, cnn, x_q, z_e, criterion, optimizer, target_class)

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

			
				##Save as pdf
				#save_query_and_gan_xp_for_final_data(I_e, cnn, z, G, z_e, original_query_img, save_name, sample_num)



				# Save explanation as txt file
				with open(path_explanations + name + '/instance_' + str(sample_num) + '_target_' + str(target_class) +'.txt', 'w') as outfile:
					np.savetxt(outfile, I_e.detach().numpy()[0][0])
				
				# New prediction of explanation x'
				new_pred = int(torch.argmax(torch.exp(cnn(I_e)[0])))
				
				# Metrics for Plausibility
				#mc_dropout_results = mc_dropout(cnn, new_pred, I_e) # Cannot use MC dropout without dropout layer
				#nn_dist, _ = k_nn.kneighbors(X=np.array(cnn(I_e)[1].detach().numpy()), n_neighbors=2) # ValueError: Incompatible dimension X shape = 1024 , Y shape = 128--> need to replicate these files but can't becuase no information is available
				#IM1 = IM1_metric(I_e, aes, original_query_pred, new_pred)
				#IM2 = IM2_metric(I_e, aes, ae_full, new_pred)

				

				temp_data = temp_data.append({'Instance': sample_num, 'Name': name, 'optim_time': optim_time, 'Original Label': original_query_label, 'Original Prediction': original_query_pred, 'Target Label': target_class, 'New Prediction': new_pred}, ignore_index=True)  #Removed: ,


				print(temp_data.head())
				expt1_data = pd.concat([expt1_data, temp_data])
		
	print(expt1_data.head())
	expt1_data.to_csv('output_MNIST_data.csv', index=False)
	print("Time to do one digit:", round(time.time() - start_time, 3))

if __name__ == "__main__":
    main()
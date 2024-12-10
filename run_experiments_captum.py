import numpy as np
from collections import OrderedDict
import captum
from captum.attr import *
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision
import argparse
from time import time
import os.path
import pathlib
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class Mnist_net_relu(torch.nn.Module):
    def __init__(self):
        super(Mnist_net_relu,self).__init__()
        self.layer1 = torch.nn.Linear(1*28*28, 1024)
        self.layer2 = torch.nn.Linear(1024, 1024)
        self.layer3 = torch.nn.Linear(1024, 1024)
        self.layer4 = torch.nn.Linear(1024, 10)


    def forward(self,x):
        x = x.view(-1, 1*28*28)
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        #features = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

class CNN_L6_128(torch.nn.Module):
	def __init__(self):
		super(CNN_L6_128, self).__init__()
		self.main = torch.nn.Sequential(
			
			# input is Z, going into a convolution
			torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
			torch.nn.BatchNorm2d(8),
			torch.nn.ReLU(True),
			torch.nn.Dropout2d(p=0.1),
			
			torch.nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(True),
			torch.nn.Dropout2d(p=0.1),

			torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(True),
			torch.nn.Dropout2d(p=0.1),

			torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(True),
			torch.nn.Dropout2d(p=0.2),

			torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(True)
		)
			
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(128, 10),
		)
		
	def forward(self, x):   
		x = self.main(x)
		x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
		logits = self.classifier(x)
		return logits


     
class AlibiCNN(torch.nn.Module):
    def __init__(self):
        super(AlibiCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)

   
        self.fc1 = nn.Linear(64 * 14 * 14, 256) 
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # Final layer without activation (will apply softmax in loss function)

        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2) # (28x28 -> 28x28)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0) # (14x14 -> 10x10)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16 channels, 5x5 feature maps
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # First convolutional layer + activation + pooling
        x = F.relu(self.conv1(x))       # Output: (6, 28, 28)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Output: (6, 14, 14)
        
        # Second convolutional layer + activation + pooling
        x = F.relu(self.conv2(x))       # Output: (16, 10, 10)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Output: (16, 5, 5)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)     # Output: (batch_size, 16*5*5)
        
        # Fully connected layers + activations
        x = F.relu(self.fc1(x))        # Output: (batch_size, 120)
        x = F.relu(self.fc2(x))        # Output: (batch_size, 84)
        x = self.fc3(x)                # Output: (batch_size, num_classes)
        
        return x


class MLP_schut(nn.Module):
    """A single multi-layer perceptron for classification."""

    def __init__(self, n_hidden=80, input_flat_size=28*28, n_classes=10):
        super().__init__()
        self._net = nn.Sequential(  #
            nn.modules.flatten.Flatten(),  #
            nn.Linear(in_features=input_flat_size, out_features=n_hidden),  #
            nn.ReLU(),  #
            nn.BatchNorm1d(num_features=n_hidden),  #
            nn.Linear(in_features=n_hidden, out_features=n_hidden),  #
            nn.ReLU(),  #
            nn.BatchNorm1d(num_features=n_hidden),  #
            nn.Linear(in_features=n_hidden, out_features=n_classes),
        )

    def forward(self, x: Tensor):
        x = self._net(x)
        probs = F.softmax(x, dim=-1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial convolution (smaller for MNIST)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Adaptive average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x







def main():
    # Add arguments
     parser = argparse.ArgumentParser(description="Experiments for Captum implementation of Counterfactual Explanations")
     parser.add_argument('--path_classifier', help='Path to the image classifier file', default='mnist_resnet8.pt')
     parser.add_argument('--dataset', help='dataset that you would like to use. Default: MNIST', default='mnist')
     parser.add_argument('--n_samples', help='Number of samples/instances you want to create explanations for', default=100) 
     parser.add_argument('--path_explanations', help='Path to output explanations and original instance', default='mnist_resnet8_output/')
     #parser.add_argument('--target_classes', help='File with target classes for each instances, if only specific classes are of interest', default='target_classes_1.csv')
     args = parser.parse_args()
     # Define arguments into variables
     path_classifier = args.path_classifier 
     dataset = args.dataset 
     n_samples = args.n_samples 
     path_explanations = args.path_explanations
     #target_classes = args.target_classes

     # Add target directory here
     save_path = '/ivi/ilps/personal/jkareem/reproducing-CFX-methods-data'


     if dataset == 'mnist':
        # Load datasets and make loaders.
        print("getting mnist...")
        completepathmnist = os.path.join(save_path, 'mnist')
        print(completepathmnist)
        if path_classifier == 'mnist_relu_4_1024.pt':
            transform = transforms.ToTensor()
        
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            
        test_set = torchvision.datasets.MNIST(root=completepathmnist, download=True, train=False, transform=transform)
        print("got data!")
        

        n_classes = 10

        ### Change model HERE
        model = ResNet(BasicBlock, [1, 1, 1], num_classes=10)
        checkpoint = torch.load(os.path.join(save_path, path_classifier), map_location='cpu')

        if path_classifier == 'mnist_relu_4_1024.pt':
            #remove constant.value
            del checkpoint['Constant.value']
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                if key.startswith('Gemm.'):
                    new_key = key.replace('Gemm', 'layer1')

                elif key.startswith('Gemm_1'):
                    new_key = key.replace('Gemm_1', 'layer2')

                elif key.startswith('Gemm_2'):
                    new_key = key.replace('Gemm_2', 'layer3')

                elif key.startswith('Gemm_3'):
                    new_key = key.replace('Gemm_3', 'layer4')

                else:
                    pass

                new_state_dict[new_key] = value
            checkpoint = new_state_dict

            model.load_state_dict(checkpoint)

        else:
            model.load_state_dict(checkpoint)
            model.eval()


     else:
        print("add alternative dataset here")
        quit()
         
     
     

     experiment_data = pd.DataFrame(columns=['Instance', 'Name', 'optim_time','Original Label', 'Original Prediction','Target Label', 'New Prediction'])




     for sample in range(n_samples):
         image, original_label = test_set[sample]
         image = image.unsqueeze(0)
         original_prediction = int(torch.argmax(torch.exp(model(image))))
         
         

         for name in ['Captum-MinParamPerturbation']:
            print(" ")
            print("-------------------------------")
            print(sample, name)
            print("-------------------------------")
            temp_data = pd.DataFrame()
            start_time = time()

            if name == 'Captum-MinParamPerturbation':
               feature_mask = torch.arange(49).reshape(7,7).repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0).reshape(1,1,28,28)
               ablator = FeatureAblation(model)
               attr = ablator.attribute(image, target=original_label, feature_mask=feature_mask)
               # Choose single channel, all channels have same attribution scores
               pixel_attr = attr[:,0:1]

               def pixel_dropout(image, dropout_pixels):
                    keep_pixels = image[0][0].numel() - int(dropout_pixels)
                    print('dropout_pixels:', int(dropout_pixels))
                    vals, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
                    return (pixel_attr < vals.item()) * image




              
               min_pert_attr = captum.robust.MinParamPerturbation(forward_func=model, attack=pixel_dropout, arg_name="dropout_pixels", mode="linear",
                                     arg_min=0, arg_max=1024, arg_step=8, apply_before_preproc=True)
               try:
                    pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(image, target=original_label, perturbations_per_eval=10)

               except RuntimeError:
                    print("all pixels have been dropped! No explanation found")
                    pixel_dropout_im = None
                    
               

            optim_time = time() - start_time

            if pixel_dropout_im == None: 
                temp_data = pd.concat([temp_data, pd.DataFrame([{'Instance': sample, 'Name': name, 'optim_time': optim_time, 'Original Label': original_label, 'Original Prediction': original_prediction, 'Target Label': None, 'New Prediction': None}])], ignore_index=True)
         
            else:
                new_prediction = int(torch.argmax(torch.exp(model(pixel_dropout_im))))
                image_tensor = torch.from_numpy(pixel_dropout_im.detach().numpy())
                temp_data = pd.concat([temp_data, pd.DataFrame([{'Instance': sample, 'Name': name, 'optim_time': optim_time, 'Original Label': original_label, 'Original Prediction': original_prediction, 'Target Label': None, 'New Prediction': new_prediction}])], ignore_index=True)
                outfile = path_explanations + name + '/instance_' + str(sample) + '_target_' + str(new_prediction) + '.pt'
                new_outfile = os.path.join(save_path, outfile)
                print(new_outfile)
                pathlib.Path(os.path.join(save_path, path_explanations + name)).mkdir(parents=True, exist_ok=True)
                torch.save(image_tensor, new_outfile)

            # Save explanation as txt file
            
            #perturbed_image = pixel_dropout_im.reshape(28, 28)
            #with open(path_explanations + name + '/instance_' + str(sample) + '_target_' + str(new_prediction) +'.txt', 'w') as outfile:
            #np.savetxt(outfile, perturbed_image.detach().numpy()
        
            
            #print("shape of image:", image_tensor.shape)
            
            print(temp_data.head())
            experiment_data = pd.concat([experiment_data, temp_data])

     print(experiment_data.head())
     csv_file_name = path_explanations +'output_Captum_'+ dataset +'_data.csv'
     new_path_csv = os.path.join(save_path,csv_file_name)
     print(new_path_csv)
     experiment_data.to_csv(new_path_csv, index=False) 





if __name__ == "__main__":
     main()

            



            







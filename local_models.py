from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F


class CNN(nn.Module):
	
	def __init__(self):
		super(CNN, self).__init__()
		self.main = nn.Sequential(
			
			# input is Z, going into a convolution
			nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			nn.Dropout2d(p=0.1),
			
			nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.Dropout2d(p=0.1),

			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Dropout2d(p=0.1),

			nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Dropout2d(p=0.2),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True)
		)
			
		self.classifier = nn.Sequential(
			nn.Linear(128, 10),
		)
		
	def forward(self, x):   
		x = self.main(x)
		x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
		logits = self.classifier(x)
		return logits, x
	

class Mnist_9_200(nn.Module):
	def __init__(self):
		# mnist_9_200
		super(Mnist_9_200, self).__init__()
		self.main = nn.Sequential(
			nn.Flatten(),
			nn.Linear(784,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU(),
			nn.Linear(200,200),
			nn.ReLU()
			# nn.ReLU(),
			# nn.Linear(10,10, bias=False)
		)

		self.classifier = nn.Sequential(nn.Linear(200,10),)

	def forward(self, x):   
		x = self.main(x)
		#x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
		logits = self.classifier(x)
		return logits, x


class Mnist_relu_4_1024(nn.Module):
	# mnist_relu_4_1024
	def __init__(self):
		super(Mnist_relu_4_1024,self).__init__()
		self.layer1 = nn.Linear(1*28*28, 1024)
		self.layer2 = nn.Linear(1024, 1024)
		self.layer3 = nn.Linear(1024, 1024)
		self.layer4 = nn.Linear(1024, 10)


	def forward(self,x):
		x = x.view(-1, 1*28*28)
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		#features = F.relu(self.layer3(x))
		logits = self.layer4(x)
		return logits, x
	


	

class AlibiCNN(nn.Module):
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
        logits = self.fc2(x)  # Final layer without activation (will apply softmax in loss function)

        return logits, x


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
        logits = self.fc3(x)                # Output: (batch_size, num_classes)
        
        return logits, x
	

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
            
        )
        self.classifier = nn.Linear(in_features=n_hidden, out_features=n_classes)

    def forward(self, x: torch.Tensor):
        x = self._net(x)
        logits = self.classifier(x)
        #probs = F.softmax(x, dim=-1)
        return logits, x
	


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
        logits = self.fc(x)

        return logits, x



	

class AE(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
			nn.ReLU(True)
		)

		self.decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),

			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ReLU(True),
			nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		x = torch.sigmoid(x)
		return x


class Generator(nn.Module):
	def __init__(self, ngpu, nc=1, nz=100, ngf=64):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
			nn.Tanh()
		)

	def forward(self, input):
		output = self.main(input)
		return output


	 
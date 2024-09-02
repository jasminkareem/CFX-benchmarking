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


	 
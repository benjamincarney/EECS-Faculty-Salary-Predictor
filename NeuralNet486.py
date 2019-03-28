import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from math import sqrt
import torch.nn as nn
import matplotlib.pyplot as plt

#Takes in csv file and loads it so that it is a pytorch tensor
def loader():
	pass


#Our Model's class
class NeuralNet(nn.module):
	def __init__(self, dim):
		super().__init__()

		#Change these to our liking. Maybe add Batchnorm or L2 Normalization?
		#Also maybe do weight initialization ourselves?

		self.embed = None #Figure out how we want to do embedding
		self.fc = nn.Sequential(
			# N x ? tensor (? WILL BE KNOWN ONCE EMBEDDING HAS BEEN IMPLEMENTED)
			nn.Linear(dim, 100),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			# N x 100 tensor
			nn.Linear(100, 200),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			# N x 200 tensor
			nn.Linear(200, 100),
			nn.ReLU(inplace=True),
			# N x 100 tensor
			nn.Linear(100,1),
			nn.ReLU(inplace=True)
			# N x 1 tensor
			)


	def forward(self, x):
		# x is an N x dim tensor
		y_hat = None #Add the embedding once figured out
		y_hat = self.fc(y_hat) 
		return y_hat


def NeuralTrain(trainloader, net, criterion, optimizer, device):
	loss_graph = []
	for epoch in range(40):  # loop over the dataset for x number of epochs
		start = time.time()
		running_loss = 0.0

		#For each batch run through model, backprop, and optimize weights
		for i, (data, salary) in enumerate(trainloader):
			data = data.to(device).float()
			salary = salary.to(device).float()

			optimizer.zero_grad()
			output = net(data)
			loss = criterion(output, salary)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			loss_graph.append(loss.item())
			if i % 100 == 99:
				end = time.time()
				print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' % 
					(epoch + 1, i + 1, running_loss / 100, end - start))
				start = time.time()
				running_loss = 0.0
    
    # Plot learning curve
	plt.title('Learning curve.')
	plt.ylabel('MSE')
	plt.xlabel('Optimization steps.')
	plt.plot(loss_list, '--')

    print('Finished Training')

def NeuralTest(testloader, net, device):
	total = 0
	with torch.no_grad():
		for data in testloader:
			representations, salary = data
			representations = representations.to(device).float()
			salary = salary.to(device).float()
			outputs = net(representations)
			error = nn.MSELoss(outputs, salary)
	print('Accuracy: %d %%' % (
        error))

def main():
	#Sets device to cpu or gpu if you have one
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	#Gets our datasets loaded
	X_train, y_train, X_test, y_test = loader()

	#Put them into torch datasets with batch size 
	#BATCH SIZE CAN CHANGE TO WHATEVER WORKS BEST
	trainset = data_utils.TensorDataset(X_train, y_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

	testset = data_utils.TensorDataset(X_test, y_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

	net = NeuralNet().to(device)
	net.init_weights()
	criterion = nn.MSELoss()
	#Can also switch from adam to sgd if we so choose
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

	train(trainloader, net, criterion, optimizer, device)
	test(testloader, net, device)



if __name__ == '__main__':
	main()



























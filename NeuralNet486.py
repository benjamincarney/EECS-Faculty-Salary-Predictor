import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Takes in csv file and loads it so that it is a pytorch tensor for training and testing data
def loader():
	X_train, y_train, X_test, y_test = None, None, None, None

	return X_train, y_train, X_test, y_test


#Our Model's class
class NeuralNet(nn.Module):
	def __init__(self, self, hidden_units1=50, hidden_units2=100, output_units=1, inp_units=20):
		super().__init__()

		#Change these to our liking. Maybe add Batchnorm or L2 Normalization?
		#Also maybe do weight initialization ourselves?

		self.embed = None #Figure out how we want to do embedding
		self.fc = nn.Sequential(
			# N x ? tensor (? WILL BE KNOWN ONCE EMBEDDING HAS BEEN IMPLEMENTED)
			nn.Linear(dim, hidden_units1),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			# N x 100 tensor
			nn.Linear(hidden_units1, hidden_units2),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			# N x 200 tensor
			nn.Linear(hidden_units2, hidden_units1),
			nn.ReLU(inplace=True),
			# N x 100 tensor
			nn.Linear(hidden_units1,output_units),
			nn.ReLU(inplace=True)
			# N x 1 tensor
			)


	def forward(self, x):
		# x is an N x dim tensor
		y_hat = None #Add the embedding once figured out
		y_hat = self.fc(y_hat) 
		return y_hat

#Set weights of model
def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)

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
	fig1, ax1 = plt.subplots()
	ax1.plot(loss_graph, '--')
	ax1.set_title('Learning curve.')
	ax1.set_ylabel('MSE')
	ax1.set_xlabel('Optimization steps.')

	print('Finished Training')

def NeuralTest(testloader, net, criterion, device):
	total = 0
	error = []
	fig2, ax2 = plt.subplots()
	with torch.no_grad():
		for data in testloader:
			representations, salary = data
			representations = representations.to(device).float()
			salary = salary.to(device).float()
			outputs = net(representations)
			loss = criterion(outputs, salary)
			ax2.plot(outputs.numpy(), salary.numpy(), 'r+')
			ax2.set_title('Prediction Plot')
			ax2.set_ylabel('Actual Salary')
			ax2.set_xlabel('Prediction')
			error.append(loss)
	print('Error: %d dollars' % (np.mean(error)))

def main():
	#Sets device to cpu or gpu if you have one
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	#Get our datasets loaded
	X_train, y_train, X_test, y_test = loader()

	
	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(X_train, y_train)

	# Make predictions using the testing set
	y_pred = regr.predict(X_test)

	# The coefficients
	print('Coefficients Lin Reg: \n', regr.coef_)
	
	# The mean squared error
	print("Mean squared error Lin Reg: %.2f" % mean_squared_error(y_test, y_pred))
	
	# Explained variance score: 1 is perfect prediction
	print('Variance score Lin Reg: %.2f' % r2_score(y_test, y_pred))
    
	# Plot outputs
	plt.plot(y_test, y_pred, 'r+')
	plt.plot(range(-5,5), range(-5,5))
	plt.show()
	

	#Put them into torch datasets with batch size 
	#BATCH SIZE CAN CHANGE TO WHATEVER WORKS BEST
	trainset = data_utils.TensorDataset(X_train, y_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

	testset = data_utils.TensorDataset(X_test, y_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

	#Model and Loss
	net = NeuralNet().to(device)
	net.apply(init_weights)
	criterion = nn.MSELoss()

	#Can also switch from adam to sgd if we so choose
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

	#Train and Test model
	NeuralTrain(trainloader, net, criterion, optimizer, device)
	NeuralTest(testloader, net, criterion, device)



if __name__ == '__main__':
	main()



























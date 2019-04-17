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
import pandas as pd

#Takes in csv file and loads it so that it is a pytorch tensor for training and testing data
def loader():
	X_train, y_train, X_test, y_test = None, None, None, None

	#Read csv file and drop unncecessary columns
	data = pd.read_csv('combined_data.csv')
	data = data.drop(data.columns[0], axis=1)
	data = data.drop(['LAST','FIRST','ID','MIDDLE','APPT FRACTION','AMT OF SALARY PAID FROM GENL FUND',
		'FOS', 'Middle', 'url', 'X1', 'found', 'at'], axis=1)

	#Get number of unique values
	columns = list(data.columns.values)
	unique_vals_set = set()
	for i in columns:
		unique_vals_set.update(set(data[i].unique().tolist()))
	unique_vals = len(unique_vals_set)

	#Turn data to numpy array
	data = data.values

	#Full dataset variables of X and Y
	y_vals = data[:, 3]
	X_vals = np.delete(data, 3, axis=1)

	indices = np.random.permutation(y_vals.shape[0])
	training_idx, test_idx = indices[:math.floor(y_vals.shape[0] * 0.9)], indices[math.floor(y_vals.shape[0] * 0.9):]
	
	X_train = X_vals[training_idx, :]
	X_test = X_vals[test_idx,:]
	y_train = y_vals[training_idx]
	y_test = y_vals[test_idx]



	return X_train, y_train, X_test, y_test, unique_vals


#Our Model's class
class NeuralNet(nn.Module):
	def __init__(self, embed_units, hidden_units1=50, hidden_units2=100, output_units=1, inp_units=20):
		super().__init__()

		#Change these to our liking. Maybe add Batchnorm or L2 Normalization?
		#Also maybe do weight initialization ourselves?

		self.embed = nn.embed(embed_units, 15) #Figure out how we want to do embedding
		self.fc = nn.Sequential(
			# N x ? tensor (? WILL BE KNOWN ONCE EMBEDDING HAS BEEN IMPLEMENTED)
			nn.Linear(15, hidden_units1),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			# N x 100 tensor
			nn.Linear(hidden_units1, hidden_units2),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			# N x 200 tensor
			nn.Linear(hidden_units2, hidden_units1),
			nn.Tanh(),
			# N x 100 tensor
			nn.Linear(hidden_units1,output_units),
			nn.ReLU(inplace=True)
			# N x 1 tensor
			)


	def forward(self, x):
		# x is an N x dim tensor
		y_hat = self.embed(x.long()) #Add the embedding once figured out
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
	X_train, y_train, X_test, y_test, unique_vals = loader()

	#Turn numpy matrices into pytorch tensors for neural network
	X_train = torch.tensor(X_train)
	y_train = torch.tensor(y_train)
	X_test = torch.tensor(X_test)
	y_test = torch.tensor(y_test)

	#Put them into torch datasets with batch size 
	#BATCH SIZE CAN CHANGE TO WHATEVER WORKS BEST
	trainset = data_utils.TensorDataset(X_train, y_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

	testset = data_utils.TensorDataset(X_test, y_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

	#Model and Loss
	net = NeuralNet(embed_units=unique_vals).to(device)
	net.apply(init_weights)
	criterion = nn.MSELoss()

	#Can also switch from adam to sgd if we so choose
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

	#Train and Test model
	NeuralTrain(trainloader, net, criterion, optimizer, device)
	NeuralTest(testloader, net, criterion, device)



if __name__ == '__main__':
	main()



























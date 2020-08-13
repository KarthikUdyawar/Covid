# python3 Covid.py

import pandas as pd
import numpy as np
df = pd.read_csv('Database.csv')

# Database

x = np.array(df.iloc[:, 0:11].values)
y = np.array(df.iloc[:, 11:12].values)

# Function

def sigmoid(x):
    if (x >= 0).all:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)
    
def d_sigmoid(x):
    return x * (1 - x)
    
def ReLu(x):
    if x<=0:
        return 0
    return x

def loss(actual, predicted):
	sum_square_error = 0.0
	for i in range(len(actual)):
		sum_square_error += (actual[i] - predicted[i])**2.0
	mean_square_error = np.round(1.0 / len(actual) * sum_square_error,5)
	return mean_square_error
    
def accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if (abs(actual[i] - predicted[i])) < 1e-2:
			correct += 1
	return correct / float(len(actual)) * 100.0

'''Nodes'''

input_nodes   = 11
hidden1_nodes = 8
hidden2_nodes = 4
output_node   = 1

'''Random seed'''

np.random.seed(1)

'''Synapses'''

syn0 = np.random.randn(input_nodes, hidden1_nodes)
syn1 = np.random.randn(hidden1_nodes,hidden2_nodes)
syn2 = np.random.randn(hidden2_nodes, output_node)

'''Learning rate & Iteration'''

bias = 1
learning_rate = 0.1
iteration = 100000

'''Training loop'''

for i in range(iteration):

    # Layers and Propagation

    l0 = x
    l1 = sigmoid(np.dot(l0, syn0) + bias)
    l2 = sigmoid(np.dot(l1, syn1) + bias)
    l3 = sigmoid(np.dot(l2, syn2) + bias)

    # Error and Back propagation

    l3_error = y - l3
    l3_delta = l3_error * d_sigmoid(l3) * learning_rate
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * d_sigmoid(l2) * learning_rate
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * d_sigmoid(l1) * learning_rate
    
    if (i % (iteration/10)) == 0:
        print('Epoch: ' + str(int(i/10000)) + '/10 -- Loss: ' + str(float(loss(y,l3))) + ' -- Acc: ' + str(float(accuracy(y,l3))))
        
    # Correction

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)

'''Print result'''

print('--------------------------------------')
print("Training complete")
#print('Actual outputs: ' + str(y.T))
#print('Output after training: ' + str(l3.T))


'''Inputs test'''

A = float(input('Inputs A : '))
B = float(input('Inputs B : '))
C = float(input('Inputs C : '))
D = float(input('Inputs D : '))
E = float(input('Inputs E : '))
F = float(input('Inputs F : '))
G = float(input('Inputs G : '))
H = float(input('Inputs H : '))
I = float(input('Inputs I : '))
J = float(input('Inputs J : '))
K = float(input('Inputs K : '))

x = np.array([A,B,C,D,E,F,G,H,I,J,K])

'''Prediction'''

l0 = x
l1 = sigmoid(np.dot(l0, syn0) + bias)
l2 = sigmoid(np.dot(l1, syn1) + bias)
l3 = sigmoid(np.dot(l2, syn2) + bias)

output = round(float(ReLu(l3)),6)
print('Predicted output: ' + str(output))

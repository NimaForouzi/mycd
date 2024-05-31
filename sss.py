import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_tensor = torch.randn(4)
###print(random_tensor)
zeros_tensor = torch.zeros(2,3)
###print(zeros_tensor)

empty_tensor = torch.empty(4,2,5, dtype=torch.double)
print(empty_tensor)
###print(empty_tensor.size())

#creating a tensor from a list datatype
my_list = [13,12.55,16,24]
list_to_tensor = torch.tensor(my_list)
###print(my_list)

#matrices(tensor in torch) operations
#1- addition by '+'
zero_matrices = torch.zeros(2,3)
one_matrices = torch.ones(2,3)
addition_operation = zero_matrices + one_matrices
###print(addition_operation)

#1- addition by a method 'add()'
addition_operation2 = torch.add(zero_matrices, one_matrices)

#1- addition and replacing in one of variables
zero_matrices.add_(one_matrices)
###print(zero_matrices)

#2- substraction by '-'
substraction_result = one_matrices - one_matrices
###print(substraction_result)

#2- substraction by method 'sub()'
substraction_result2 = torch.sub(one_matrices, one_matrices)
###print(substraction_result2)

#2- substract and replace
one_matrices.sub_(one_matrices)
###print(one_matrices)

one_matrices = torch.ones(2,3)
#-3 multiplication (elementwise / not matrics multiplication)
mul_result = torch.mul(one_matrices, one_matrices)
mul_result2 = one_matrices * one_matrices
one_matrices.mul_(one_matrices)
###print(mul_result)

#-4 division
division_result = torch.div(one_matrices, one_matrices)
division_result2 = one_matrices / one_matrices
one_matrices.div_(one_matrices)
###print(one_matrices)

""""""""""""""""""""""""""
#slicing the tesnor
random_tensor2 = torch.rand(5,3)
###print(random_tensor2)
#all rows - just first column
###print(random_tensor2[: ,0])
#first row - all columns
###print(random_tensor2[0, :])
""""""""""""""""""""""""
#resizing the tensors
random_tensor3 = torch.rand(2,4)
#turining it into a 1D vector with 8 elements
reshaped_to_1D = random_tensor3.view(8)
###print(reshaped_to_1D.size())

#if we just know number of rows OR column of new reshaped matrices, we put '-1' for row or column number which we don't know
reshaped_to_only_we_know_column_is_2 = random_tensor3.view(-1,2)
#(automatically system place '4' instead of '-1' since it is just possible dimension)
###print(reshaped_to_only_we_know_column_is_2.size())

#turning a tensor to a numpy array
random_tensor4 = torch.ones(4).to(device)
#OR
random_tensor = torch.ones(4, device=device)

###print(random_tensor4)
turn_into_numpy_array = random_tensor4.to("cpu").numpy()
####print(type(turn_into_numpy_array))
###print(type(random_tensor4))

#if our tensor be also on CPU (in my case it is not), by any change on tensor, the numpy array also changes
random_tensor4.add_(1)
###print(turn_into_numpy_array)
###print(random_tensor4)
###print(turn_into_numpy_array)
# no problem was occured since tensor is on GPU but numpy array is on CPU

""""""""""""""""""
#from numpy to tensor
random_numpy_array = np.ones(6)
turn_to_tensor = torch.from_numpy(random_numpy_array)
###print(type(turn_to_tensor))

#change from 'gpu' to cpu
random_tensor4 = random_tensor4.to("cpu")
###print(random_tensor4.device)
""""""""""""""""""""""""""""""""
#if we know that our tensor is going to undergone optimization algoritms, so we use 'requires_grad=True' while defining the tensor
need_optimization_tensor = torch.ones(8, requires_grad=True)
###print(need_optimization_tensor)
""""""""""""""""""""""""""""""
#calculating the gradient
random_tensor5 = torch.randn(5, requires_grad=True)
###print(random_tensor5)
#imagine random_tensor5 as x (input)
y = random_tensor5 + 2
###print(y)
z = y*y*2
###print(z)
z = z.mean()
###print(z)
z.backward() #dz/dx
###print(random_tensor5.grad) #gradient vector 
""""""""""""""""""""""""
#our output was a real number in previous example, if our outpus is a vector, we should pass a vector with same size to the backward()
z = y*y+2
vector_made = torch.tensor([1,1,1,0,0])
z.backward(vector_made)
###print(random_tensor5.grad)
###print(random_tensor5)
""""""""""""""
#if we want our updating weights be departed from gradient calculation, we can use three methods
#none of the below requires gredient
#random_tensor5.requires_grad_(False)
##print(random_tensor5)
#random_tensor5.detach()
with torch.no_grad():
    newRandom = random_tensor5
 #   ##print(newRandom)

""""""""""""""""""
#you should turn zero the gradient each iteration for showing right gradients

weights = torch.ones(5, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    ##print(weights.grad)
    weights.grad.zero_()

#or if we want to do optimization
#note: when using optim and passing the weights, you should pass the variable weights in a bracket
optimizer = torch.optim.SGD([weights], lr = 0.01)
optimizer.step()
optimizer.zero_grad()

""""""""""""
#perform a forward pass and backpropogation
#forward pass with input x=1, y=2, w=1 initially
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
y_hat = w*x
squared_loss = (y-y_hat)**2
##print(squared_loss)
squared_loss.backward()
##print(w.grad)

""""""""""""""""""""
#lets calculate the linear regression manually using numpy arrays
# lets imagine our model is f and f = x * 2
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

#model calculation
def forward(x):
    return w*x

#lets say loss function is MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#gredient of MSE is 1/N * 2x * (wx - y)
def gredient(x,y,y_prediction):
    return np.dot(2*x,y_prediction-y).mean()

n_iteration = 20
lr= 0.01

#for a random entery
##print("prediction before the training is", forward(5))   

for epoch in range(n_iteration):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gredient(X,Y,y_pred)
    w -= lr*dw
    
    #for each iteration show it
    #if epoch % 1 == 0:
        ##print('for epoch number',epoch+1,": the w is ",w ,"loss is:",l)

#prediction for input 7 should be 14
##print("prediction after the training is", forward(7)) 

""""""""""""""""""""""""""""""
#lets calculate the linear regression using torch
# lets imagine our model is f and f = x * 2
X1 = torch.tensor([1,2,3,4],dtype=torch.float32)
Y1 = torch.tensor([2,4,6,8], dtype=torch.float32)

w1 = torch.tensor([0.0], dtype =torch.float32, requires_grad=True)

#model calculation
def forward(x):
    return w1*x

#lets say loss function is MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


n_iteration = 20
lr= 0.01

#for a random entery
##print("prediction before the training is", forward(5))   

for epoch in range(n_iteration):
    y_pred = forward(X1)
    l = loss(Y1, y_pred)
    l.backward()
    with torch.no_grad():
        w1 -= lr*w1.grad
    w1.grad.zero_()
    #for each iteration show it
    #if epoch % 1 == 0:
        ##print('for epoch number',epoch+1,": the w is ",w ,"loss is:",l)

#prediction for input 7 should be 14
##print("prediction after the training is", forward(7)) 
""""""""""""""
#doing everything in torch way
#note: each row is a data sample in torch. so each [n] is a row with 1 column (1 feature)
X2 = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y2 = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([7], dtype=torch.float32)

#you don't have to define variable for 'w' weights while you are using the torch
n_samples, n_features,  = X2.shape
#print(n_samples, n_features)
number_of_input_feature = number_of_output_feature = n_features
number_of_epoches = 20
lr1 = 0.01
import torch.nn as nn
forward_by_torch = nn.Linear(number_of_input_feature, number_of_output_feature)

#loss
loss_value = nn.MSELoss()
#gradient calculater:
optimizer = torch.optim.SGD(forward_by_torch.parameters(), lr=lr1)
#print(f"the prediction before training the model is:{forward_by_torch(X_test).item():.3}" )
#gradient
for epoch in range (number_of_epoches):
    y_predict = forward_by_torch(X2)
    lossval = loss_value(Y2, y_predict)
    lossval.backward()
    optimizer.step()
    optimizer.zero_grad()
    [w, b] = forward_by_torch.parameters()
    #print(f"for epoch number: {epoch+1} the w is{w[0][0].item():0.3f} and the loss is {lossval:0.3f}")
    #################################################################

    #making the model (linear regression in previous example from scratch)
X1 = torch.tensor([1,2,3,4],dtype=torch.float32)
Y1 = torch.tensor([2,4,6,8], dtype=torch.float32)

w1 = torch.tensor([0.0], dtype =torch.float32, requires_grad=True)

#model calculation
def forward(x):
    return w1*x

#lets say loss function is MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


n_iteration = 20
lr= 0.01

#for a random entery
##print("prediction before the training is", forward(5))   

for epoch in range(n_iteration):
    y_pred = forward(X1)
    l = loss(Y1, y_pred)
    l.backward()
    with torch.no_grad():
        w1 -= lr*w1.grad
    w1.grad.zero_()
    #for each iteration show it
    #if epoch % 1 == 0:
        ##print('for epoch number',epoch+1,": the w is ",w ,"loss is:",l)

#prediction for input 7 should be 14
##print("prediction after the training is", forward(7)) 
""""""""""""""
#doing everything in torch way
#note: each row is a data sample in torch. so each [n] is a row with 1 column (1 feature)
X3 = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y3 = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test1 = torch.tensor([7], dtype=torch.float32)

#you don't have to define variable for 'w' weights while you are using the torch
n_samples, n_features = X3.shape
#print(n_samples, n_features)
number_of_input_feature = number_of_output_feature = n_features
number_of_epoches = 20
lr1 = 0.01
import torch.nn as nn
"""forward_by_torch = nn.Linear(number_of_input_feature, number_of_output_feature)"""
class myLinearRegression(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.lin = nn.Linear(input_dimension, output_dimension)
    def forward(self, x):
        return self.lin(x)
forward_by_torch2 = myLinearRegression(number_of_input_feature, number_of_output_feature)
#loss
loss_valued = nn.MSELoss() #is equal to manual function and take 2 argument
#gradient calculater:
optimizer1 = torch.optim.SGD(forward_by_torch2.parameters(), lr=lr1)
#print(f"the prediction before training the model is:{forward_by_torch(X_test).item():.3}" )
#gradient
for epoch in range (number_of_epoches):
    y_predict = forward_by_torch2(X3)
    lossval = loss_valued(Y3, y_predict)
    lossval.backward()
    optimizer1.step() #is euqal to w -= lr*w.gradient
    optimizer1.zero_grad() #is euqal to w.gradient.zero()
    w, b = forward_by_torch2.parameters()
    print(f"for epoch number: {epoch+1} the w is{w[0][0].item():0.3f} and the loss is {lossval:0.3f}")

##################################################
#for working with torch
#0- prepare the data
#1- design the model
#2- define the loss and optimizer
#3 write the training loop

import torch
import torch.nn as nn
import sklearn.datasets as datasets
import matplotlib as plt

x_numpy, y_numpy = datasets.make_regression()

#x_numpy, y_numpy = datase

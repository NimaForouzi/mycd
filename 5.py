import numpy as np
import torch 
import torch.nn as nn

#softmax take logits (raw output of previous layer) and make probabilities between 0 to 1
#softmax formula: e^y / sum(e^y)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

data_point = np.array([2.0, 1.0, 0.1])
outputs = softmax(data_point)
#print(outputs)

#implement the softmax function in torch
x = torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0)

#CROSS-ENTROPY LOSS (FOR CLASSIFICATION)
def cross_entropy_loss(target_y, predicted_y):
    return -(np.sum((target_y * np.log(predicted_y)))) #it should be devided by target_label.shape[0] (number of data)

#example:
#these target_y is one-hot encoded (meaning one class (true class) has 1.0 and others 0.0)
target_y = np.array([1.0, 0.0, 0.0])

#following values are after implementing the softmax which generate probabilities between 0 to 1
good_predicted_y = np.array([0.7, 0.2, 0.1])
bad_predicted_y = np.array([0.1, 0.3, 0.6])

loss_for_good_prediction = cross_entropy_loss(target_y=target_y, predicted_y=good_predicted_y)
loss_for_bad_prediction = cross_entropy_loss(target_y=target_y, predicted_y=bad_predicted_y)
print(f"{loss_for_good_prediction:4f}")
print(f"{loss_for_bad_prediction:.4f}")

#implementation in torch
#CrossEntropyLoss() automatically applies softmax and after cross entropy loss
#NOTICE: for using CrossEntropyLoss() our target label should NOT be one-hot encoded and
#values for y_target should be real values(e.g. [0.4, 0.2, 0.4]) not one-hot encoded(e.g. [1.0, 0.0, 0,0])
#NOTICE: y_prediction for nn.CrossEntropyLoss() should be logit (raw scores) without applying softmax before
cross_entropy_torch = nn.CrossEntropyLoss()
#following line says the true class if first class ([0])
y_target_torch = torch.tensor([0])
print(y_target_torch.item())
#NOTICE: for this you should have n_samples * n_class, in this example we have 1*3 (one sample three classes)
#NOTICE: these are logit(raw scores) before applying the softmax, so 2.0 will be bigger probability
good_prediction_torch = torch.tensor([[2.0, 1.0, 0.1]]) #[[]] meaning that we have only 1 data (1 row)[[],[]] meaning we have two data
bad_prediction_torch = torch.tensor([[0.5, 4.0, 0.3]])

#since the tensor has single value we cam call item()
print(f"good loss:{cross_entropy_torch(good_prediction_torch, y_target_torch).item()}")
print(f"bad loss:{cross_entropy_torch(bad_prediction_torch, y_target_torch).item()}")
"""y_target_torch = torch.tensor([0, 1])  # First data point belongs to class 0, second data point belongs to class 1

# Define predictions for two data points
good_prediction_torch = torch.tensor([[2.0, 1.0, 0.1], [2.0, 3.0, 1.0]])
NOTICE: we have n_samples precited_y, so when we have one predicted class means we have just 1 data point
and good loss actually have 3 predctios that this data belongs to class 0, or 1 or 2 which in our example belongs
to class 0 so higher logit (which later softmax will be applied) shows higher confidance
"""

#we had three prediction for three classes in two models (one bad one good), now if we want to see actual
#prediction of each model as final prediction we can do this:
# 1 means find maximum along columns (meaning compare all columns related to each row)
# if 1 was 0, mean that compare all rows in a specifid column

maximum_value_from_each_row_data, pridction_model_1 = torch.max(good_prediction_torch, 1)
maximum_value_from_each_row_data1, pridction_model_2 = torch.max(bad_prediction_torch, 1)
#print(f"good:{maximum_value_from_each_row_data, pridction_model_1}")
#print(f"bad:{maximum_value_from_each_row_data1, pridction_model_2}")
#showing that model one predict class 0 while model two predict class 1

#lets say we have multiple data point
multiple_target = torch.tensor([0, 2, 1])
#meaning: first data point class 0 is correct, second data point class 2 and third datapoint class 1

#first data should higher logit for class 0, second higher logit for class 2 and so on
good_model = torch.tensor([[5.0, 0.2, 0.5], [0.2, 0.5, 4.0], [0.4, 9.0, 0.3]])
bad_model = torch.tensor([[0.1, 3.3, 0.5], [0.3, 4.5, 0.1], [3.0, 0.1, 3.0]])
mul_loss1 = cross_entropy_torch(good_model, multiple_target)
mul_loss2 = cross_entropy_torch(bad_model, multiple_target)
print(f"Cross entropy for good model{mul_loss1:.4f}")
print(f"Cross entropy for bad model{mul_loss2:.4f}")

#implement a simple neural network

#for classification number of n_output is number of classes we have in general (dog, cat, etc.)
class neural_network_1(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(neural_network_1, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out = self.linear1(x)   
        out = self.relu(out)
        out = self.linear2(out)
        #no softmax since we use torch.CrossEntropyLoss
        return out 
    
model = neural_network_1(400000,5,3)    
criterion = nn.CrossEntropyLoss()
#if want use BineryCrossEntropy (just two class)

class two_class_network(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(two_class_network, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        #we have just one output (neuron) which predict the picture is or is not the specified class
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out=self.linear2(out)
        out = nn.Sigmoid(out)
        return out    
    
model_2 = two_class_network(233, 2)
#BineryCrossEntropyLoss needs sigmoid after last layer in torch while 
criterion2 = nn.BCELoss()
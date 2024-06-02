# LOGISTIC REGRESSION USING PYTORCH
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1) Prepare the data
# make the dataset by scikit learn
data_set = datasets.load_breast_cancer() # breast cancer dataset is for a logistic regression dataset
x,y = data_set.data, data_set.target

#print(x.shape, y.shape) 
n_samples, n_features = x.shape

# we want 20% our data be 'test data'
#random_state any number, so our data be reproducable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=4321412)

#scale our data
sc =StandardScaler() # standardize features by removing the mean and scaling to unit variance. 
#Standardizing data is a common preprocessing step before feeding it into many (turn into normal distribution)
#machine learning algorithms, as it ensures that each feature contributes equally to the model.
#This means it transforms the data such that the distribution of each feature has a 
#mean of 0 and a standard deviation of 1. Feature scaling is a crucial step in the preprocessing of data
# for machine learning models, especially for algorithms that are sensitive to the magnitudes of the features,
# such as Support Vector Machines (SVM) and k-Nearest Neighbors (k-NN).
"""StandardScaler is sensitive to outliers because it uses the mean and standard deviation, which can be heavily influenced by extreme values. For data with significant outliers, alternative scalers like RobustScaler may be more appropriate."""
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#turn default numpy to turn and from double to float
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#turn the y to a column vector
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


"""2) model"""
#logistic regression:
#first a linear wx+b and then sigmoid
class logstic_regression(nn.Module):
    def __init__(self, n_input_features):
        super(logstic_regression, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = logstic_regression(n_features)
"""3) loss and optimization"""
lr = 0.01
#Binary Cross-Entropy Loss(BCE) is a performance measure for classification models 
#that outputs a prediction with a probability value typically between 0 and 1
criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
"""4) training loop"""
n_iteration = 100
for epoch in range (n_iteration):
    #forward pass
    y_prediciton = model(x_train)
    #backward
    loss = criterion(y_prediciton, y_train)
    loss.backward()

    #update the weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f"epoch: {epoch+1} the loss is {loss.item():.4f}")

#evaluation of our model
        
#evaluation should no be part of our calculation so no_grad
with torch.no_grad():
    y_prediction_for_test = model(x_test)
    #the output of foward fun is a number between 0 and 1 (less than 0.5 count 1 and
    #more then 0.5 count as 1, so)
    y_prediction_for_test_class = y_prediction_for_test.round()
    """we didn't want to round the value for x_train because we don't care about
    values of x_train, they are just for training the weights"""
    #y_prediction_for_test_class.eq(y_test).sum() -> if class prediction is equal to class y_test
    #add a plus 1 (+1) 
    accuracy = y_prediction_for_test_class.eq(y_test).sum() / y_test.shape[0] #number of test data (number of rows of y_test)
    print(f"our accuracy is {accuracy:.4f}")
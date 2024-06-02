"""epoch -> one forward and backward training of ALL training dataset"""
"""batch -> number of training samples for one forward and backward pass"""
"""number of iterations -> number of passes, which in each pass there is number of a batch"""
"""e.g. -> 100 datapoint, batchsize=20 so -> n_iteration=5 for 1 epoch, n_iteration 10 for 2 epoch"""
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
#class wineDataset(Dataset)->inside pranthesis means ineherentence related to OOP
class wineDataset(Dataset):  
    def __init__(self):
        #loading data
        #delimiter=',' -> our data is sepearated by ','
        #skiprows=1 -> first row is header (name of the features) so we skip that
        #we want later use torch so np.float32
        x_y = np.loadtxt("/home/vboxuser/mycd/wine.csv", delimiter=',', skiprows=1, dtype=np.float32)
        #x_y is 2D so we use [#:# , #,#]\
        # we want ALL rows (all data) and all column exept column 1 (which is label y)
        # so for x we have x_y[:,1:]
        self.x = torch.from_numpy(x_y[:,1:]) 
        #for y we want again all data (all rows) but just first column (labels)
        self.y = torch.from_numpy(x_y[:,[0]]) 
        #n_samples is number of rows which is in shape pf x_y
        self.n_samples = x_y.shape[0]

    def __getitem__(self, index):
        #allow indexing
        #this returns a tuple containig features of a datapoint and its label
        return self.x[index], self.y[index]
    def __len__(self):
        #allow call length of our dataset
        return self.n_samples

dataset = wineDataset()
# we can directly use instance 'dataset' and summen dataset[0] because there is a method
#called __getitem__ which automatically allow use instance to access the index of data
#so dataset[0] will summen first x[index] and y[index] which is x[0] and y[0] and bring back a tuple related to x and y
first_data_pont = dataset[0]
#since first_data_point is a tuple of x[index] and y[index] automatically add x[index] to feature_values and y[index] to label_of_first_data
features_values_of_first_data, label_of_first_data = first_data_pont
#print(f"the first data (row) feature values are:{features_values_of_first_data}")
#print(f"the label is{label_of_first_data}")

#loading the data, batchsize=4 (only load 4 data point)
#shuffle=True -> shuffle the data for loading
#num_worker -> number of process 
dataloader = DataLoader(dataset = dataset,  batch_size=4, num_workers=2, shuffle=True)

#aaa = next(iter(loade))
#feature_values, lables = aaa
#print(f"features are {feature_values} and lables are {lables}")

n_epoch = 2
total_number_samples = len(dataset)
batch_size = 4
n_iteration = math.ceil(total_number_samples/batch_size)
print(f"num samples {total_number_samples} and number of iteration {n_iteration}")

for epoch in range(n_epoch):
    # i is index and also inputs and lables are unpack from dataloader
    #enumerate() says that do the unpacking (because of 'in' keword) on each data in dataloader instead of unpacking
    #on the whole dataloader which contain lots of data
    for i, (input,lables) in enumerate(dataloader):
        #dummy training
        #forward
        if(i+1) % 5 == 0:
            print(f"epoch {epoch+1}/{n_epoch}, step {i+1}/{n_iteration}, input{input.shape}")
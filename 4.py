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
    def __init__(self,  transform = None):
       
        x_y = np.loadtxt("/home/vboxuser/mycd/wine.csv", delimiter=',', skiprows=1, dtype=np.float32)
       
        self.x = x_y[:,1:]
        
        self.y = x_y[:,[0]]
        #n_samples is number of rows which is in shape pf x_y
        self.n_samples = x_y.shape[0]
        self.transform = transform

    #allow to directly instance[indexing][indexing]
    def __getitem__(self, index):
        #allow indexing
        #this returns a tuple containig features of a datapoint and its label
        sample = self.x[index], self.y[index]
        if (self.transform):
            sample = self.transform(sample)
        return sample
    
    #alow use len(name_of_instance) to find the length
    def __len__(self):
        #allow call length of our dataset
        return self.n_samples



#create our on transform class

#a transform to transform to tensor
class ToTensor:
    #like __len__ and __getitem__, this function also allow a streight use of 'instance name' for a goal
    #by __call__ we can instance_name(arugment1, argument2) and codes inside __call__ can be runned
    def __call__(self, sample):
        inputs, target_labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(target_labels)
    
dataset = wineDataset(transform=ToTensor())
first_data = dataset[0]
feature_values, labels = first_data
#print(type(feature_values), type(labels))

#Lets make another transform
class my_multiplication_transform:
    #we want argument when an nstance is made, so we create a constructor
    def __init__(self, factor):
        self.factor = factor

    #we want use instane of this class as a function later and give some argument (in addition to primary one for constructor)
    # so we shoudl use __call__ class to use it as a functiom
    def __call__(self, datapoint):
        x_cordinates, y_coordinates = datapoint
        x_cordinates *= self.factor
        return x_cordinates, y_coordinates

#lets say we want to implement both previous created transform on the dataset
composed = torchvision.transforms.Compose([ToTensor(),my_multiplication_transform(2)])   
newDataset = wineDataset(transform=composed)
firstOne = newDataset[0] #because of __getitem__ function
feature_value_for_first_data, label = firstOne
print(feature_values) #for just one transform
print(feature_value_for_first_data) #for two transform   
# both of our transforms (turn into tensor) and also (multipli by two) is working here
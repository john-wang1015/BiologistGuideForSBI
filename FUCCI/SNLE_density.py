import numpy as np
import pandas as pd
import torch
_ = torch.manual_seed(10)
import os
import math
from sbi import utils as utils
import sbi
from sbi import inference
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import scipy.io as sio
import matlab.engine

eng = matlab.engine.start_matlab()

class CustomPriorDist:
    def __init__(self, mean,sigma, return_numpy: bool = False):
        self.dist1 = utils.BoxUniform(mean,sigma)
        self.dist2 = utils.BoxUniform(mean,sigma)
        self.dist3 = utils.BoxUniform(mean,sigma)
        self.dist4 = utils.BoxUniform(mean,sigma)
        self.dist5 = utils.BoxUniform(mean,sigma)
        self.dist6 = utils.BoxUniform(mean,sigma)        
        self.return_numpy = return_numpy
        
    def sample(self, sample_shape=torch.Size([])):
        if len(sample_shape) == 1:
            length = sample_shape[0]
            samples = torch.ones(length,6)
            temp_1 = self.dist1.sample(sample_shape)
            temp_2 = self.dist2.sample(sample_shape)
            temp_3 = self.dist3.sample(sample_shape)
            temp_4 = self.dist4.sample(sample_shape)
            temp_5 = self.dist5.sample(sample_shape)
            temp_6 = self.dist6.sample(sample_shape)
            samples[:,0] = temp_1[:,0]
            samples[:,1] = temp_2[:,0]
            samples[:,2] = temp_3[:,0]
            samples[:,3] = temp_4[:,0]
            samples[:,4] = temp_5[:,0]
            samples[:,5] = temp_6[:,0]
            return samples.numpy() if self.return_numpy else samples
        else:
            samples = torch.ones(1,6)
            temp_1 = self.dist1.sample(sample_shape)
            temp_2 = self.dist2.sample(sample_shape)
            temp_3 = self.dist3.sample(sample_shape)
            temp_4 = self.dist4.sample(sample_shape)
            temp_5 = self.dist5.sample(sample_shape)
            temp_6 = self.dist6.sample(sample_shape)

            samples[:,0] = temp_1[0]
            samples[:,1] = temp_2[0]
            samples[:,2] = temp_3[0]
            samples[:,3] = temp_4[0]
            samples[:,4] = temp_5[0]
            samples[:,5] = temp_6[0]
            return samples.numpy() if self.return_numpy else samples            
    
    def log_prob(self, values):
        log_probs = torch.ones((values.size()[0],))
        length = values.size()[0]
        if self.return_numpy:
            values = torch.as_tensor(values)
        
        for i in range(values.size()[0]):
            temp = torch.ones(6)
            temp[0] = self.dist1.log_prob(values[i][0])
            temp[1] = self.dist2.log_prob(values[i][1])
            temp[2] = self.dist3.log_prob(values[i][2])
            temp[3] = self.dist4.log_prob(values[i][3])
            temp[4] = self.dist5.log_prob(values[i][4])
            temp[5] = self.dist6.log_prob(values[i][5])
            log_probs[i] =  torch.sum(temp)
            
        return log_probs.numpy() if self.return_numpy else log_probs
    

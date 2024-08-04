# -*- coding: utf-8 -*-
"""
implicit U-Net enhanced Fourier Neural Operator
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import scipy.io
import os

torch.manual_seed(123)
np.random.seed(123)


################################################################
# fourier layers

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = min(modes4, 3//2+1)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,z,t ), (in_channel, out_channel, x,y,z,t) -> (batch, out_channel, x,y,z,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] =self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

class U_net(nn.Module):  
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate): 
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate) 
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):  
        batchsize, width = x.shape[0], x.shape[1]
        out_conv1 = self.conv1(x.view(batchsize, width, -1))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))   
        out_conv3 = self.conv3_1(self.conv3(out_conv2)) 
        
        out_deconv2 = self.deconv2(out_conv3) 
        concat2 = torch.cat((out_conv2, out_deconv2), 1) 
        out_deconv1 = self.deconv1(concat2) 
        concat1 = torch.cat((out_conv1, out_deconv1), 
        out_deconv0 = self.deconv0(concat1)  
        concat0 = torch.cat((x, out_deconv0.view(batchsize, width, 32, 33, 16, 3)), 1)  
        out = self.output_layer(concat0.view(concat0.shape[0], concat0.shape[1], -1)).view(batchsize, width, 32, 33, 16, 3)  
        
        return out   

    def conv(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.LeakyReLU(0.1, inplace=True),  
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=4,
                                stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2)


class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width, nlayer): 
        super(FNO4d, self).__init__()

        """
        input: the solution of the first 5 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        
        output: the solution of the next  timestep
        
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.fc0 = nn.Linear(9, self.width)  # 5+4 (u(1, x, y), ..., u(5, x, y), x,y,z,t)
        self.nlayer = nlayer
        
        self.convlayer = nn.ModuleList([SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4).cuda() for i in range(1)])
        self.w = nn.ModuleList([nn.Conv1d(self.width, self.width, 1).cuda() for i in range(1)])
        self.u = nn.ModuleList([U_net(self.width, self.width, 3, 0).cuda() for i in range(1)])
        
        self.fc1 = nn.Linear(self.width, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):   
        batchsize = x.shape[0]
        size_x, size_y, size_z, size_w = x.shape[1], x.shape[2], x.shape[3], x.shape[4] 

        grid = self.get_grid(batchsize, size_x, size_y, size_z, size_w, x.device)
        x = torch.cat((x, grid), dim=-1) 

        x = self.fc0(x)   
        x = x.permute(0, 5, 1, 2, 3, 4) 
        coef = 1./self.nlayer

        for i in range(self.nlayer-1):
            
            x1 = self.convlayer[0](x) 
            x2 = self.w[0](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
            x3 = self.u[0](x-x1)
            x = F.gelu(x1+x2+x3)*coef + x  

        
        x1 = self.convlayer[0](x)
        x2 = self.w[0](x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        x3 = self.u[0](x-x1)
        x = (x1+x2+x3)*coef + x
      
        x = x.permute(0, 2, 3, 4, 5, 1) 
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x  

    def get_grid(self, batchsize, size_x, size_y, size_z, size_w, device ): 
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, size_w, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).repeat([batchsize, size_x, 1, size_z, size_w, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).repeat([batchsize, size_x, size_y, 1, size_w, 1])
        gridw = torch.tensor(np.linspace(0, 1, size_w), dtype=torch.float)
        gridw = gridw.reshape(1, 1, 1, 1, size_w, 1).repeat([batchsize, size_x, size_y, size_z, 1, 1])

        return torch.cat((gridx, gridy, gridz, gridw), dim=-1).to(device) #




# configs
################################################################

device = torch.device("cuda")
#-------------------------------------------------------------------------------
#tunning3
modes = 8
width = 80
epochs = 100
learning_rate = 0.001
weight_decay_value = 1e-11
nlayer = 40

#---------------------------------------------------------------------------------------------

batch_size = 4
scheduler_step = 10
scheduler_gamma = 0.5 

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()


################################################################
# load data
################################################################


#-------------------------------------------------
vor_data = np.load('../../fno_data/data_mave.npy') #
vor_data = vor_data[...,0:3]
vor_data = vor_data[0:20,...]


vor_data = torch.from_numpy(vor_data) #
print(vor_data.shape)


input_list = []
output_list = []

for j in range(vor_data.shape[0]):
    for i in range(vor_data.shape[1]-5):
        
        input_list.append(vor_data[j,i:i+5,...])
        output_6m5 = (vor_data[j,i+5,...]-vor_data[j,i+4,...])
        output_list.append(output_6m5) 
                     
### switch dimension
      
input_set = torch.stack(input_list) 
output_set = torch.stack(output_list) 
input_set = input_set.permute(0,2,3,4,5,1) 


full_set = torch.utils.data.TensorDataset(input_set, output_set)
train_dataset, test_dataset = torch.utils.data.random_split(full_set, [int(0.8*len(full_set)), 
                                                                       len(full_set)-int(0.8*len(full_set))])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO4d(modes, modes, modes, modes, width, nlayer).to(device)


print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

mse_train = []
mse_test = []


myloss = LpLoss()

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    for xx, yy in train_loader:
        
        xx = xx.to(device)
        yy = yy.to(device)
        im = model(xx).to(device)
        
        train_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    mse_train.append(train_loss.item())
        

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            im = model(xx).to(device)
            test_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))
        mse_test.append(test_loss.item())

    t2 = default_timer()
    
    
    print(ep, "%.2f" % (t2 - t1), 'train_loss: {:.4f}'.format(train_loss.item()), 
          'test_loss: {:.4f}'.format(test_loss.item()))

MSE_save=np.dstack((mse_train,mse_test)).squeeze()
np.savetxt('./loss_IUFNO.dat',MSE_save,fmt="%16.7f")

torch.save(model.state_dict(), 'weights_IUFNO.pth') 






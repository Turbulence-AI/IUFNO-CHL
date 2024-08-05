# IUFNO-CHL
The code is contained in iufno_code.zip.  
The datasets will be available at https://www.kaggle.com/datasets/aifluid/coarsened-fdns-data-chl soon.   
1. The code is for the implicit U-Net enhanced Fourier Neural Operator. 
2. Pytorch and scipy are required for running this code.
3. The dataset for training and testing can be downloaded at https://www.kaggle.com/datasets/aifluid/coarsened-fdns-data-chl
4. The shape of the dataset is 21x400x32x33x16x4 for Re=180,21x400x64x49x32x4 for Re=395 and 21x400x64x65x32x4 for Re=590. Here 21 is the number of groups with the first 20 groups for training and the last 1 group for testing. 400 represents 400 time steps. 32x33x16 (64x49x32 and 64x65x32) represents the shape of the grids. 4 represents (u,v,w,p) and here p is not used for the current incompressible flow.
5. In the datasets, the "*mave*" files are the fluctuation part of the flow field and the "*ave*" files are the mean field which is averaged along the homogeneous x and z directions, the t direction, and amoung the 20 groups. Hence, the shape of the "*ave*" files is 1x1x1x33x1x4 for Re=180, etc. 
Note that the "*ave*" files are only needed in the testing process when the predicted fluctuations are added back to the mean field. The reader is referred to the following paper for details:
https://arxiv.org/abs/2403.03051

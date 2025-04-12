#!/usr/bin/python3.8
# @Author  : Tianshu Zheng
# @Email   : zhengtianshu996@gmail.com
# @File    : train.py
# @Software: PyCharm

import sys
import argparse
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from net import SparseReconstruction
from net import Mapping
from vit_pytorch import ViT
from Dataform import Mydataset
import math
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=25230, help="size of the batches") # 841 for MRI image 
parser.add_argument(
    "--dataset",
    type=str,
    default="./image/data.mat",
    help="root directory of the dataset",
)
parser.add_argument(
    "--input_nc", type=int, default=60, help="number of channels of input data"
)
parser.add_argument(
    "--output_nc", type=int, default=3, help="number of channels of output data"
)
parser.add_argument(
    "--cuda", action="store_false", default="False", help="use GPU computation"
)
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use")
parser.add_argument("--net_sp", type=str, default="./attempt/sp.pth")
parser.add_argument("--Mapping", type=str, default="./attempt/mapping.pth")
parser.add_argument("--v", type=str, default="./attempt/v.pth")
parser.add_argument(
    "--size1", type=int, default=3, help="size of the data crop (squared assumed)"
)
parser.add_argument(
    "--mask", type=str, default="./example/mask.nii", help="mask used for generation"
)
parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
net_sp = SparseReconstruction()

v = ViT(
    image_size=3,
    patch_size=3,
    num_classes=60 * 1 * 1,
    dim=512,
    depth=8,
    heads=10,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
)

Mapping = Mapping()

if opt.cuda:
    net_sp#.cuda()
    v#.cuda()
    Mapping#.cuda()

# Loss function
criterion = nn.MSELoss()

# Optimizers
optimizer_v = optim.Adam(v.parameters(), lr=opt.lr)
optimizer_sp = optim.Adam(net_sp.parameters(), lr=opt.lr)
optimizer_mapping = optim.Adam(Mapping.parameters(), lr=opt.lr)

# Inputs & targets memory allocation
# Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
Tensor = torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size1, opt.size1)
input_B = Tensor(opt.batchSize, opt.output_nc, 1)

tau = Tensor(1, 1).fill_(1e-10)

# Dataset loader
testset = Mydataset(opt.dataset)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=opt.batchSize,
    shuffle=True,
)

# Training loop
for epoch in range(opt.epochs):
    running_loss = 0.0
    for i, batch in enumerate(testloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch["A"]))
        real_B = Variable(input_B.copy_(batch["B"]))

        # Zero the parameter gradients
        optimizer_v.zero_grad()
        optimizer_sp.zero_grad()
        optimizer_mapping.zero_grad()

        # Forward pass
        t = v(real_A)
        t = t.reshape(real_A.shape[0], real_A.shape[1], 1, 1)
        dataconv = t + (real_A[:, :, 1, 1]).reshape(
            real_A.shape[0], real_A.shape[1], 1, 1
        )
        same_B = net_sp(dataconv)
        same_B = same_B.reshape(same_B.shape[0], same_B.shape[1])
        Viso = (same_B[:, -1]).reshape(same_B.shape[0], 1)
        Vf = (same_B[:, :-1]).reshape(same_B.shape[0], -1)
        Vf = (Vf + tau) / torch.norm((Vf + tau), p=1, dim=1).reshape(Vf.shape[0], 1)
        Vf = Vf.unsqueeze(2).unsqueeze(3)
        sameVf = Mapping(Vf)
        Vic = (sameVf[:, 0, :, :]).squeeze(2)
        ODI = (sameVf[:, 1, :, :]).squeeze(2)
        same_B4 = torch.cat([Vic, Viso, ODI], dim=1)

        # Compute the loss
        loss = criterion(same_B4, real_B.squeeze(2))

        # Backward pass and optimization
        loss.backward()
        optimizer_v.step()
        optimizer_sp.step()
        optimizer_mapping.step()

        running_loss += loss.item()

    # Print statistics
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(testloader)}")

print("Finished Training")

# Save the trained models
torch.save(v.state_dict(), opt.v)
torch.save(net_sp.state_dict(), opt.net_sp)
torch.save(Mapping.state_dict(), opt.Mapping)
    
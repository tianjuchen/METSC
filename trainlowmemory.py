#!/usr/bin/python3.8
# @Author  : Tianshu Zheng
# @Email   : zhengtianshu996@gmail.com
# @File    : train.py
# @Software: PyCharm

import argparse, sys
from torch.autograd import Variable
import torch
import torch.optim as optim
import numpy as np
from net import SparseReconstruction
from net import Mapping
from vit_pytorch import ViT
from Dataform import Mydataset
import nibabel as nib
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import gc

# Set smaller batch size and add gradient accumulation
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=512, help='size of the batches')  # Reduced from 20184
parser.add_argument('--accumulation_steps', type=int, default=4, help='gradient accumulation steps')
parser.add_argument('--dataset', type=str, default='./image/data.mat', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=60, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', default=False, help='use GPU computation')  # Changed default to True
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use')  # Reduced from 8
parser.add_argument('--net_sp', type=str, default='./model/sp.pth')
parser.add_argument('--Mapping', type=str, default='./model/mapping.pth')
parser.add_argument('--v', type=str, default='./model/v.pth')
parser.add_argument('--size1', type=int, default=3, help='size of the data crop (squared assumed)')
parser.add_argument('--mask', type=str, default='./example/mask.nii', help='mask used for generation')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--save_interval', type=int, default=10, help='save model every n epochs')
parser.add_argument('--train', action='store_true', default=True, help='train the model')

opt = parser.parse_args()
print(opt)


# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)
# Set device
# device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

# Networks with smaller dimensions
net_sp = SparseReconstruction().to(device)
v = ViT(
    image_size=3,
    patch_size=1,
    num_classes=60 * 1 * 1,
    dim=128,  # Reduced from 256/512
    depth=4,  # Reduced from 6
    heads=4,  # Reduced from 8
    mlp_dim=128,  # Reduced from 256/512
    dropout=0.1,
    emb_dropout=0.1
).to(device)
Mapping = Mapping().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizers with gradient clipping
optimizer_sp = optim.Adam(net_sp.parameters(), lr=opt.lr)
optimizer_v = optim.Adam(v.parameters(), lr=opt.lr)
optimizer_mapping = optim.Adam(Mapping.parameters(), lr=opt.lr)

# Load state dicts if not training
if not opt.train:
    net_sp.load_state_dict(torch.load(opt.net_sp, map_location=device))
    Mapping.load_state_dict(torch.load(opt.Mapping, map_location=device))
    v.load_state_dict(torch.load(opt.v, map_location=device))

    net_sp.eval()
    Mapping.eval()
    v.eval()

# Dataset loader with pin_memory if using GPU
pin_memory = (device.type == 'cuda')
dataset = Mydataset(opt.dataset)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=pin_memory,
    persistent_workers=True
)

testloader = DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=opt.n_cpu,
    pin_memory=pin_memory,
    persistent_workers=True
)

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def train(epoch):
    net_sp.train()
    v.train()
    Mapping.train()
    
    epoch_loss = 0
    start_time = time.time()
    optimizer_sp.zero_grad()
    optimizer_v.zero_grad()
    optimizer_mapping.zero_grad()
    
    for i, batch in enumerate(trainloader):
        # Move data to device
        real_A = batch['A'].to(device, non_blocking=True)
        real_B = batch['B'].to(device, non_blocking=True)
        
        # Forward pass
        t = v(real_A)
        t = t.reshape(real_A.shape[0], real_A.shape[1], 1, 1)
        dataconv = t + (real_A[:, :, 1, 1]).reshape(real_A.shape[0], real_A.shape[1], 1, 1)
        same_B = net_sp(dataconv)
        same_B = same_B.reshape(same_B.shape[0], same_B.shape[1])
        
        Viso = (same_B[:, -1]).reshape(same_B.shape[0], 1)
        Vf = (same_B[:, :-1]).reshape(same_B.shape[0], -1)
        Vf = (Vf + torch.tensor(1e-10, device=device)) / torch.norm((Vf + torch.tensor(1e-10, device=device)), p=1, dim=1).reshape(Vf.shape[0], 1)
        Vf = Vf.unsqueeze(2).unsqueeze(3)
        sameVf = Mapping(Vf)
        
        Vic = (sameVf[:, 0, :, :]).squeeze(2)
        ODI = (sameVf[:, 1, :, :]).squeeze(2)
        same_B4 = torch.cat([Vic, Viso, ODI], dim=1)
        
        # Calculate loss with gradient accumulation
        loss = criterion(same_B4, real_B.squeeze(2)) / opt.accumulation_steps
        loss.backward()
        epoch_loss += loss.item() * opt.accumulation_steps
        
        # Step optimizers only after accumulation steps
        if (i + 1) % opt.accumulation_steps == 0 or (i + 1) == len(trainloader):
            torch.nn.utils.clip_grad_norm_(net_sp.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(Mapping.parameters(), max_norm=1.0)
            
            optimizer_sp.step()
            optimizer_v.step()
            optimizer_mapping.step()
            
            optimizer_sp.zero_grad()
            optimizer_v.zero_grad()
            optimizer_mapping.zero_grad()
            
            cleanup()
    
    epoch_loss /= len(trainloader)
    print(f'[Epoch {epoch}] Loss: {epoch_loss:.4f} | Time: {time.time()-start_time:.2f}s')
    cleanup()
    return epoch_loss

def test():
    net_sp.eval()
    v.eval()
    Mapping.eval()
    
    store = []
    test_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            real_A = batch['A'].to(device, non_blocking=True)
            real_B = batch['B'].to(device, non_blocking=True)
            
            t = v(real_A)
            t = t.reshape(real_A.shape[0], real_A.shape[1], 1, 1)
            dataconv = t + (real_A[:, :, 1, 1]).reshape(real_A.shape[0], real_A.shape[1], 1, 1)
            same_B = net_sp(dataconv)
            same_B = same_B.reshape(same_B.shape[0], same_B.shape[1])
            
            Viso = (same_B[:, -1]).reshape(same_B.shape[0], 1)
            Vf = (same_B[:, :-1]).reshape(same_B.shape[0], -1)
            Vf = (Vf + torch.tensor(1e-10, device=device)) / torch.norm((Vf + torch.tensor(1e-10, device=device)), p=1, dim=1).reshape(Vf.shape[0], 1)
            Vf = Vf.unsqueeze(2).unsqueeze(3)
            sameVf = Mapping(Vf)
            
            Vic = (sameVf[:, 0, :, :]).squeeze(2)
            ODI = (sameVf[:, 1, :, :]).squeeze(2)
            same_B4 = torch.cat([Vic, Viso, ODI], dim=1)
            
            loss = criterion(same_B4, real_B.squeeze(2))
            test_loss += loss.item()
            
            output = same_B4.detach().cpu().numpy()
            store.append(output)
            cleanup()
    
    test_loss /= len(testloader)
    print(f'[Test] Loss: {test_loss:.4f}')
    
    if not opt.train:
        result1 = np.concatenate(store, axis=0)
        mask_nii = nib.load(opt.mask)
        mask = mask_nii.get_fdata()
        affine = mask_nii.affine
        
        for ii in range(3):
            data = result1[:,ii].reshape(mask.shape, order='F') * mask
            image = nib.Nifti1Image(data, affine)
            nib.save(image, './output/result'+str(ii)+'.nii.gz')
    
    cleanup()
    return test_loss

if __name__ == '__main__':

    try:
        if opt.train:
            best_loss = float('inf')
            
            for epoch in range(1, opt.epochs + 1):
                train_loss = train(epoch)
                test_loss = test()
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(net_sp.state_dict(), './image/trainres/best_sp.pth')
                    torch.save(v.state_dict(), './image/trainres/best_v.pth')
                    torch.save(Mapping.state_dict(), './image/trainres/best_mapping.pth')
                
                if epoch % opt.save_interval == 0:
                    torch.save(net_sp.state_dict(), f'./image/trainres/sp_epoch{epoch}.pth')
                    torch.save(v.state_dict(), f'./image/trainres/v_epoch{epoch}.pth')
                    torch.save(Mapping.state_dict(), f'./image/trainres/mapping_epoch{epoch}.pth')
        else:
            test()
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        cleanup()
        sys.exit(1)
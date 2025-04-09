import os
import torch
import torch.backends.cudnn as cudnn 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator
from model import Generator

torch.manual_seed(0)
device = torch.device("cuda:0")
cudnn.benchmark = True
mode = "train"
exp_name = "exp000"

if mode == "train":
    dataset_dir = "data"
    batch_size  = 128
    
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    
    start_epoch = 0
    resume = False
    resume_d_weight = ""
    resume_g_weight = ""
    
    epochs = 128
    
    criterion = nn.BCELoss().to(device)
    
    d_optimizer = optim.Adam(discriminator.parameters(), 0.0002, (0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), 0.0002, (0.5, 0.999))

    writer = SummaryWriter(os.path.join("samples", "logs", exp_name))
    
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)

elif mode == "valid":
    exp_dir = os.path.join("results", "test", exp_name)
    model = Generator().to(device)
    model_path = f"samples/{exp_name}/g-last.pth"

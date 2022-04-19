import numpy as np
import torch
import sys
import time

from torch.utils.tensorboard import SummaryWriter

import config
import model
from data_utils import get_dataset

def train_model(cfg, device, writer):
    #initalize model
    cnn = model.ConvNet(cfg).to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=cfg.learning_rate)

    #get data  
    train_loader = get_dataset("../../data/cnn/human/chr21.tfrecord")
    valid_loader = get_dataset("../../data/cnn/human/chr22.tfrecord")
    print("Data Loaded")

    #train model
    cnn.train_model(train_loader, valid_loader, device, criterion, optimizer, writer, cfg)
   
    #save model 
    torch.save(cnn.state_dict(), cfg.model_dir + cfg.model_name + '.pth')
    

if __name__ == '__main__':
    model_name = sys.argv[1]
    cfg = config.Config(model_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print(device)

    #set up tensorboard logging
    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter('../../data/cnn/tensorboard_logs/' + model_name + '_' + timestr)

    train_model(cfg, device, writer)
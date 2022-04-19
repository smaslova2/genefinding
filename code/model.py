import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import random
import numpy as np
import copy
from tqdm import tqdm


# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, cfg):
        super(ConvNet, self).__init__()
        self.num_filters = cfg.num_filters
        self.num_classes = cfg.num_classes

        self.max_activations = torch.zeros(self.num_filters)
        self.ave_activations = torch.zeros(self.num_filters)
        self.pfms = torch.zeros(self.num_filters, 4, 22)
        self.influence = torch.zeros(self.num_filters)

        ##################################
        ### First Layer
        ##################################
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, 
                      out_channels=196,
                      kernel_size=(1, 22),
                      stride=(1,1), 
                      padding=(0,0)),  # padding is done in forward method along 1 dimension only
            nn.BatchNorm2d(196),
            nn.Dropout(0.05))
        
        self.layer1_process = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,2)),
            nn.Dropout(p=0.05))
 
        
        ##################################
        ### Convolutional Layers
        ##################################
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=196,
                      out_channels=235,
                      kernel_size=6,
                      stride=1,
                      padding=3),
            nn.BatchNorm1d(235),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2), 
            nn.Dropout(p=0.05))
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=235,
                      out_channels=282,
                      kernel_size=6,
                      stride=1,
                      padding=3),
            nn.BatchNorm1d(282),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(p=0.05))
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=282,
                      out_channels=338,
                      kernel_size=6,
                      stride=1,
                      padding=3),
            nn.BatchNorm1d(338), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(p=0.05))

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=338,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm1d(384), 
            nn.ReLU(),
             nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(p=0.05))

        ##################################
        ### Dilated Convolutions
        ##################################           
        
        self.layer6 = nn.Sequential(
            nn.Conv1d(in_channels=338,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=2, 
                      dilation = 2),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))

        self.layer7 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=4, 
                      dilation = 4),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))
        self.layer8 = nn.Sequential(
            nn.Conv1d(in_channels=64*2,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=8, 
                      dilation = 8),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
           # nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))
        self.layer9 = nn.Sequential(
            nn.Conv1d(in_channels=64*3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=16, 
                      dilation = 16),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))
        self.layer10 = nn.Sequential(
            nn.Conv1d(in_channels=64*4,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=32, 
                      dilation = 32), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))
        self.layer11 = nn.Sequential(
            nn.Conv1d(in_channels=64*5,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=64, 
                      dilation = 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))
        self.layer12 = nn.Sequential(
            nn.Conv1d(in_channels=64*6,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=128, 
                      dilation = 128),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=(0,1)),
            nn.Dropout(p=0.05))
        
        self.layer13 = nn.Sequential(
            nn.Conv1d(in_channels=64*7, #338, #
                      out_channels=200,
                      kernel_size=3,
                      stride=1,
                      padding=1, 
                      dilation = 1), 
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(p=0.1)
        )
        self.layer14a = nn.Sequential(
            nn.Linear(in_features=2200, 
                      out_features = 1000), 
                      nn.ReLU(),
                      nn.Dropout(p=0.03))
        self.layer14 = nn.Sequential(
            nn.Linear(in_features=1000,  
                      out_features=self.num_classes))


    def forward(self, input):
        # run all layers on input data
        # add dummy dimension to input (for num channels=4)
        input = torch.reshape(input, (input.shape[0], 4, -1))
        input = torch.unsqueeze(input, 2)

        # Convolutional layers
        input = F.pad(input, (11, 11), mode='constant', value=0) # padding - last dimension goes first

        out = self.layer1_conv(input)
        out = self.layer1_process(out)
        
        out = torch.squeeze(out)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Densely connected dilated convoultions
        out = self.layer6(out)
        tmp = self.layer7(out)
        out = torch.cat((out,tmp),dim=1)
        tmp = self.layer8(out)
        out = torch.cat((out,tmp),dim=1)
        tmp = self.layer9(out)
        out = torch.cat((out,tmp),dim=1)
        tmp = self.layer10(out)
        out = torch.cat((out,tmp),dim=1)
        tmp = self.layer11(out)
        out = torch.cat((out,tmp),dim=1)   
        tmp = self.layer12(out)
        out = torch.cat((out,tmp),dim=1)

        # final layers
        out = self.layer13(out)

        out = out.view(out.shape[0], -1)
        out = self.layer14a(out)
        predictions = self.layer14(out)

        return predictions


    def train_model(self, train_loader, valid_loader, device, criterion, optimizer, writer, cfg):
        num_epochs = cfg.num_epochs
        output_directory = cfg.output_directory
        #total_step = len(train_loader)

        #open files to log error
        train_error = open(output_directory + "training_error.txt", "a")
        valid_error = open(output_directory + "valid_error.txt", "a")

        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss_valid = float('inf')
        best_epoch = 1
        i=0
        for epoch in range(num_epochs):
            running_loss = 0.0
            for(seqs, labels) in tqdm(train_loader):
                i+=1
                seqs = seqs.to(device)
                labels = labels.to(device)

                # Forward pass
                pred = self.forward(seqs)
                loss = criterion(pred, labels) # change input to 
                running_loss += loss.item()
            
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('training loss', loss, i)
                

            #save training loss to file
            epoch_loss = running_loss / len(train_loader.dataset)
            print("%s, %s" % (epoch, epoch_loss), file=train_error)

            #calculate validation loss for epoch
            valid_loss = 0.0
            with torch.no_grad():
                self.eval()
                for i, (seqs, labels) in enumerate(valid_loader):
                    seqs = seqs.to(device)
                    labels = labels.to(device)
                    pred = self.forward(seqs)
                    loss = criterion(pred, labels)
                    valid_loss += loss.item() 

            valid_loss = valid_loss / len(valid_loader.dataset)
            writer.add_scalar('validation loss', valid_loss, epoch)

            #save outputs for epoch
            print("%s, %s" % (epoch, valid_loss), file=valid_error)

            if valid_loss < best_loss_valid:
                best_loss_valid = valid_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(self.state_dict())
                print ('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}' 
                        .format(epoch+1, best_loss_valid))


        train_error.close()
        valid_error.close()

        self.load_state_dict(best_model_wts)


    def test(self, test_loader, device):
        num_filters=self.layer1_conv[0].out_channels
        emb_size=self.layer14[0].out_features
        predictions = torch.empty(0, emb_size).to(device)
        target_values = torch.empty(0, emb_size).to(device)

        with torch.no_grad():
            self.eval()
            for i, (seqs, labels) in enumerate(tqdm(test_loader)):
                seqs = seqs.to(device)
                labels = labels.to(device)
                target_values = torch.cat((target_values, labels), 0)

                pred = self.forward(seqs)
                predictions = torch.cat((predictions, pred), 0) 


        predictions = predictions.cpu().detach().numpy()
        target_values = target_values.cpu().detach().numpy() 

        return predictions, target_values

    def get_activations(self, test_loader, device):
        with torch.no_grad():
            self.eval()
            for i, (seqs, labels) in enumerate(tqdm(test_loader)):
                seqs = seqs.to(device)
                labels = labels.to(device)

                # Forward pass
                seqs= torch.unsqueeze(seqs, 2)

                # Run convolutional layers
                seqs = F.pad(seqs, (11, 11), mode='constant', value=0) 
                out = self.layer1_conv(seqs)
        
        return out

    def ave_activation_hook(self):
        def hook(module, input, output):
            batch_max = torch.max(output.reshape(output.shape[1], -1), dim=1)
            max_activations = torch.max(self.max_activations, batch_max.values.cpu().detach())
            self.max_activations = max_activations
        
        return hook

    def pfm_hook(self, nseqs):
        def hook(module, input, output):
            for filt in range(self.num_filters):
                idx = torch.where(torch.squeeze(output[:,filt,:]) > 0.5*self.max_activations[filt])
                nseqs[filt] += idx[0].shape[0]

                seqs = torch.squeeze(input[0])
                for seq in range(idx[0].shape[0]):
                    start = idx[1][seq]
                    stop = start+22
                    self.pfms[filt, :, :] += seqs[idx[0][seq], :, start:stop].cpu().detach()

        return hook

    def get_motifs(self, test_loader, device):
        #get average activations for each filter across all samples
        h1 = self.layer1_conv.register_forward_hook(self.ave_activation_hook())
        out = self.get_activations(test_loader, device)
        h1.remove()
        
        #print(self.max_activations)

        nseqs = torch.zeros(self.num_filters)
        h2 = self.layer1_conv.register_forward_hook(self.pfm_hook(nseqs))
        out = self.get_activations(test_loader, device)
        h2.remove()

        return self.pfms.cpu().detach().numpy() 

    def influence_hook(self, filt):
        def hook(module, input, output):
            ave_act = torch.mean(output[:, filt, ...])
            output[:, filt, ...] = torch.full(size=output[:, filt, ...].shape, fill_value=ave_act)
            
            return output

        return hook
        
    def get_influence(self, test_loader, device):
        num_filters=self.layer1_conv[0].out_channels

        with torch.no_grad():
            self.eval()
            for i, (seqs, labels) in enumerate(tqdm(test_loader)):
                seqs = seqs.to(device)
                labels = labels.to(device)
                pred = self.forward(seqs)
                mse = nn.MSELoss(reduction='none')
                loss = mse(pred, labels)

                for filt in range(num_filters):
                    h1 = self.layer1_conv.register_forward_hook(self.influence_hook(filt))
                    pred2 = self.forward(seqs)
                    loss2 = mse(pred2, labels)
                    #print(loss.get_device())
                    #print(loss2.get_device())
                    #print(self.influence.get_device())

                    self.influence[filt] += torch.sum((loss-loss2)**2).detach().cpu()
                    h1.remove()

        num_samples = len(test_loader.dataset)
        self.influence = self.influence/num_samples

        return self.influence


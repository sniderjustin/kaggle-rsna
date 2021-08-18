"""Take the average of each image type. Use single averaged image as input for the model."""

import os
import json
import glob
import random
import collections

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
from sklearn.metrics import roc_auc_score, roc_curve, auc

import time

import torch
from torch import nn
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import efficientnet_pytorch

from sklearn.model_selection import StratifiedKFold

def load_dicom(path):
    # input is file path
    dicom = pydicom.read_file(path)  # read data
    data = dicom.pixel_array  # to np array
    data = data - np.min(data)  # normalize
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)  # make 0 to 255 int for image
    return data  # return np int array

def visualize_sample(
    brats21id,  # patient id
    slice_i,  # image index (0.5=middle)
    mgmt_value,  # target value cancerous or not
    types=("FLAIR", "T1w", "T1wCE", "T2w")  # image types
):
    plt.figure(figsize=(16, 5))  # create figure
    patient_path = os.path.join(  # create file path
        "data/train/",
        str(brats21id).zfill(5),  # pad with zeros
    )
    for i, t in enumerate(types, 1):  # for 4 scan types
        t_paths = sorted(  # sorted list of paths
            glob.glob(os.path.join(patient_path, t, "*")), # get scans in file
            key=lambda x: int(x[:-4].split("-")[-1]),  # sort in numerical order
        )
        data = load_dicom(t_paths[int(len(t_paths) * slice_i)])  # get int array
        plt.subplot(1, 4, i)  # create sub plot 1 by 4
        plt.imshow(data, cmap="gray")  # add image to plot
        plt.title(f"{t}", fontsize=16)  # add subplot title
        plt.axis("off")  # no axis

    plt.suptitle(f"MGMT_value: {mgmt_value}", fontsize=16)
    plt.show()

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class DataRetriever(torch_data.Dataset):
    def __init__(self, paths, targets):
        self.paths = paths  # Id values
        self.targets = targets  # Target MGMT values
          
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        _id = self.paths[index]  # Get ID
        patient_path = f"data/train/{str(_id).zfill(5)}/"  # Get path
        channels = []
        for t in ("FLAIR", "T1w", "T1wCE", "T2w"): # "T2w"  # Cycle through data types
            t_paths = sorted(  # Get sorted list of file paths
                glob.glob(os.path.join(patient_path, t, "*")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            # start, end = int(len(t_paths) * 0.475), int(len(t_paths) * 0.525)
            x = len(t_paths)  # Get number of paths
            if x < 10:
                r = range(x)  # Set range of paths
            else:
                d = x // 10  # Sample every 10th image
                r = range(d, x - d, d)
                
            channel = []
            # for i in range(start, end + 1):
            for i in r:  # Loop through images. Downsample, normalize, and append. 
                channel.append(cv2.resize(load_dicom(t_paths[i]), (256, 256)) / 255)
            channel = np.mean(channel, axis=0)  # take average
            channels.append(channel)  # append average
            
        y = torch.tensor(self.targets[index], dtype=torch.float)  # Targets to tensor
        
        return {"X": torch.tensor(channels).float(), "y": y}  # provide X and y

# Model extentends nn.Module
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # use EfficientNet
        self.net = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
        # Load weights
        # checkpoint = torch.load("../input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        # self.net.load_state_dict(checkpoint)
        # Get number of features in model
        n_features = self.net._fc.in_features
        # Add fully connected layer going from efficientnet to one out value
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
    
    def forward(self, x):
        out = self.net(x)
        return out

class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg

        
class AccMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        # incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg

class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion, 
        loss_meter, 
        score_meter
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter
        
        self.best_valid_score = -np.inf
        self.n_patience = 0
        
        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
            "patience": "\nValid score didn't improve last {} epochs."
        }
    
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        # Write to tensorboard
        writer = SummaryWriter()
        # Visualize model graph
        writer.add_graph(self.model, next(iter(train_loader))["X"].to(self.device))

        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_score, train_time = self.train_epoch(train_loader)
            valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader)
            
            writer.add_scalar('Loss/train', train_loss, n_epoch)
            writer.add_scalar('Loss/valid', valid_loss, n_epoch)
            writer.add_scalar('Accuracy/train', train_score, n_epoch)
            writer.add_scalar('Accuracy/valid', valid_score, n_epoch)

            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_time
            )
            
            self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_time
            )

            if True:
#             if self.best_valid_score < valid_score:
                self.info_message(
                    self.messages["checkpoint"], self.best_valid_score, valid_score, save_path
                )
                self.best_valid_score = valid_score
                self.save_model(n_epoch, save_path)
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break
        writer.close()
            
    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()
        
        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)
            
            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())

            self.optimizer.step()
            
            _loss, _score = train_loss.avg, train_score.avg
            message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}'
            self.info_message(message, step, len(train_loader), _loss, _score, end="\r")
        
        return train_loss.avg, train_score.avg, int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)
                
            _loss, _score = valid_loss.avg, valid_score.avg
            message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}'
            self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")
        
        return valid_loss.avg, valid_score.avg, int(time.time() - t)
    
    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


if __name__=='__main__':
    set_seed(42)

    # Get labels
    # Split into train and validation sets
    train_df = pd.read_csv('data/train_labels.csv')
    df = pd.read_csv("data/train_labels.csv")
    df_train, df_valid = sk_model_selection.train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_df["MGMT_value"],
    )

    # Create dataset instances
    train_data_retriever = DataRetriever(
        df_train["BraTS21ID"].values, 
        df_train["MGMT_value"].values, 
    )

    valid_data_retriever = DataRetriever(
        df_valid["BraTS21ID"].values, 
        df_valid["MGMT_value"].values,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_retriever = DataRetriever(
        df_train["BraTS21ID"].values, 
        df_train["MGMT_value"].values, 
    )

    valid_data_retriever = DataRetriever(
        df_valid["BraTS21ID"].values, 
        df_valid["MGMT_value"].values,
    )

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
    )

    model = Model()
    model.to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001)
    criterion = torch_functional.binary_cross_entropy_with_logits

    trainer = Trainer(
        model, 
        device, 
        optimizer, 
        criterion, 
        LossMeter, 
        AccMeter
    )

    history = trainer.fit(
        100000, 
        train_loader, 
        valid_loader, 
        f"best-model-0.pth", 
        100,
    )
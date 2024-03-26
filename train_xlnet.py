import pandas as pd
from utils import loader
from models import xlnet
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from dataset import get_train_test_loaders

if __name__ == "__main__":
    torch.set_warn_always(False)

    train_loader, test_loader, num_labels = get_train_test_loaders("xlnet", num_files=4)


    EPOCHS=3

    print("Training model...")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    model = xlnet.train_model(train_loader, 
                              test_loader, 
                              num_labels,
                              EPOCHS=EPOCHS,
                              TRAIN_SIZE=0.7,
                              FREEZE=True
                            )
    print("Model trained.")

    PATH = f"trained_models/XLNet-cased-{EPOCHS}"
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")
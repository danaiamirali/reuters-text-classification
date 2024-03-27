from transformers import BertTokenizer, BertModel
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from .helpers import freeze_model as freeze_bert
import pandas as pd
import numpy as np
from torch import cuda
from sklearn.model_selection import train_test_split

device = 'cuda' if cuda.is_available() else 'cpu'

"""
This file is used to train a BERT model on the Reuters dataset.

The model will perform multi-label classification, 
predicting the topics of each document.

"""


# Creating the customized model, by adding a drop out and a dense layer on top of bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self, NUM_LABELS):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-cased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, NUM_LABELS)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        # print(output_1.shape)
        output_2 = self.l2(output_1)
        # print(output_2.shape)
        output = self.l3(output_2)
        # print(output.shape)
        return output

def train_model(training_loader: DataLoader, 
                testing_loader: DataLoader,
                NUM_LABELS: int,
                MAX_LEN: int = 200,
                EPOCHS: int = 1,
                LEARNING_RATE: float = 1e-05,
                TRAIN_SIZE: float = 0.8,
                THRESHOLD: float = None,
                FREEZE: bool = True
    ) -> BERTClass:
    """
    Main driver function to train the BERT model.
    """

    model = BERTClass(NUM_LABELS)
    model.to(device)

    if FREEZE:
        model = freeze_bert(model)
        print("BERT model frozen.")

    def train(epoch):
        model.train()
        for _,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            ids = ids.squeeze(1)
            mask = mask.squeeze(1)
            token_type_ids = token_type_ids.squeeze(1)
            outputs = model(ids, mask, token_type_ids)
        
            optimizer.zero_grad()

            # This should not modify anything but confirm the expectation
            assert outputs.shape == targets.shape, "Mismatch in output and target shapes"

            loss = loss_fn(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            loss.backward()
            optimizer.step()

    def validation(epoch):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                ids = ids.squeeze(1)
                mask = mask.squeeze(1)
                token_type_ids = token_type_ids.squeeze(1)
                # print("val output shape1", ids.shape, mask.shape, token_type_ids.shape, targets.shape)
                outputs = model(ids, mask, token_type_ids)

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}...")
        train(epoch)
        if THRESHOLD == None:
            candidate_thresholds = [0.1 + 0.05*i for i in range(1, 6)]
        else:
            candidate_thresholds = [THRESHOLD]
        outputs, targets = validation(epoch)
        for THRESHOLD in candidate_thresholds:
            print("THRESHOLD:", threshold)    
            outputs = np.array(outputs) >= THRESHOLD
            print("Number of labels predicted:", np.sum(outputs))
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro', zero_division=np.nan)
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro', zero_division=np.nan)
            # clf_report = metrics.classification_report(targets, outputs, zero_division=np.nan)
            print(f"Validation Accuracy Score = {accuracy}")
            
            try: 
                hs = hamming_score(targets, outputs)
                print(f"Validation Hamming Score = {hs}")
            except:
                pass
            
            print(f"Validation F1 Score (Micro) = {f1_score_micro}")
            print(f"Validation F1 Score (Macro) = {f1_score_macro}")
            # print(clf_report)

        torch.save(model.state_dict(), f"checkpoints/BERT-cased-{epoch}")

    return model
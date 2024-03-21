from transformers import BertTokenizer, BertModel
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
import regex as re
import pandas as pd
import numpy as np
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

"""
This file is used to train a BERT model on the Reuters dataset.

The model will perform multi-label classification, 
predicting the topics of each document.

"""
def preprocess(data: str) -> torch.Tensor:
    if not isinstance(data, str):
        print(type(data))
    # we will keep casing
    # replace numbers
    data = re.sub(r"\d+", "<NUM>", data)
    return data

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.body = dataframe.body
        self.targets = self.data.topics
        self.max_len = max_len

    def __len__(self):
        return len(self.body)

    def __getitem__(self, index):
        body = str(self.body[index])
        body = " ".join(body.split())

        inputs = self.tokenizer.encode_plus(
            body,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self, NUM_LABELS):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
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

def train_model(df: pd.DataFrame, 
                MAX_LEN: int = 200,
                TRAIN_BATCH_SIZE: int = 8,
                VALID_BATCH_SIZE: int = 4,
                EPOCHS: int = 1,
                LEARNING_RATE: float = 1e-05,
                TRAIN_SIZE: float = 0.8,
                NUM_LABELS: int = None
    ) -> BERTClass:
    """
    Main driver function to train the BERT model.

    Assumes that the dataframe has the following columns:
        - body: The body of the document.     (preprocessed, not tokenized, not padded)
        - topics: The topics of the document. (multi-label, binarized format)
    """

    if NUM_LABELS is None:
        NUM_LABELS = df["topics"].shape[1]
        print("Inferred NUM_LABELS from dataframe as: ", NUM_LABELS)

    model = BERTClass(NUM_LABELS)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_dataset=df.sample(frac=TRAIN_SIZE,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

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
            # print("train output shape1", ids.shape, mask.shape, token_type_ids.shape, targets.shape)
            outputs = model(ids, mask, token_type_ids)
            # print("train output shape2", outputs.shape)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
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
                # print("val output shape2", outputs.shape)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(epoch)

        outputs, targets = validation(epoch)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")

        torch.save(model.state_dict(), "checkpoints/BERT-cased-{epoch}")

    return model
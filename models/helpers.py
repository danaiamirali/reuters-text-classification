import torch
from torch.utils.data import Dataset
import regex as re
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.body = dataframe.body
        self.targets = np.array([topic for topic in dataframe.topics])
        self.max_len = max_len

    def __len__(self):
        return len(self.body)

    def __getitem__(self, index):
        body = str(self.body.iloc[index])
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
    
def freeze_model(model):
    for param in model.l1.parameters():
        param.requires_grad = False

    return model

def freeze_layers(model, num_layers):
    for i in range(num_layers):
        for param in model.l1.encoder.layer[i].parameters():
            param.requires_grad = False

    return model

def preprocess(data: str) -> torch.Tensor:
    if not isinstance(data, str):
        print(type(data))
    # we will keep casing
    # replace numbers
    data = re.sub(r"\d+", "<NUM>", data)
    return data
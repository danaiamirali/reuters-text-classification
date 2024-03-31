from transformers import BertModel
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from train_common import freeze_layers, eval_metrics, find_optimal_thresholds
import numpy as np
from torch import cuda

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
                num_labels: int,
                epochs: int = 1,
                learning_rate: float = 1e-04,
                freeze_num: int = 1,
                print_metrics: bool = True,
                print_thresholds: bool = True
    ) -> BERTClass:
    """
    Main driver function to train the BERT model.
    """

    model = BERTClass(num_labels)
    model.to(device)

    model = freeze_layers(model, freeze_num)

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

            loss = loss_fn(outputs, targets, compute_class_weights(targets))
            if print_metrics:
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
                # get model outputs
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    # func to compute inverse class frequency weights given a set of labels
    def compute_class_weights(labels: torch.Tensor | np.ndarray):
        weights = np.bincount(labels.flatten().astype(int))
        weights[weights == 0] = 1
        weights = 1 / weights
        weights = weights / weights.sum()
        return torch.Tensor(weights)
    
    def loss_fn(outputs, targets, weights: torch.Tensor = None):
        assert outputs.shape == weights.shape, "Weights of incorrect shape"
        return torch.nn.BCEWithLogitsLoss(weight=weights)(outputs, targets)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch}...")
        train(epoch)

        candidate_thresholds = [0 + 0.0125 * i for i in range(70)]
        outputs, targets = validation(epoch)
        optimal_thresholds = find_optimal_thresholds(targets, 
                                                     outputs, 
                                                     lambda x, y : metrics.f1_score(x, y, average="macro", zero_division=np.nan), 
                                                     candidate_thresholds, 
                                                     num_labels)

        if print_thresholds:
            for num, threshold in enumerate(optimal_thresholds, 0):
                print(f"Label {num} : Threshold = {threshold}")

        if print_metrics:
            print("---- Validation Metrics ----")
            eval_metrics(targets, outputs, optimal_thresholds)

        torch.save(model.state_dict(), f"checkpoints/BERT-cased-{epoch}")

    return model
from models import bert
import torch
import os
from utils import config
from dataset import get_train_test_loaders

if __name__ == "__main__":
    torch.set_warn_always(False)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

    train_loader, test_loader, num_labels = get_train_test_loaders("bert", num_files=2)

    EPOCHS = config("train.epochs")
    lr = config("train.lr")
    print("Training model...")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    model = bert.train_model(train_loader, 
                              test_loader, 
                              num_labels,
                              epochs=EPOCHS,
                              freeze_num=3,
                              learning_rate=lr
                            )
    print("Model trained.")

    PATH = f"trained_models/BERT-cased-{EPOCHS}"
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")
  
def freeze_model(model):
    for param in model.l1.parameters():
        param.requires_grad = False

    return model

def freeze_layers(model, num_layers):
    for i in range(num_layers):
        for param in model.l1.encoder.layer[i].parameters():
            param.requires_grad = False

    return model


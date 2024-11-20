def freeze_model(model):
    """Disables gradient calculation for params of a given torch model

    Parametres:
        model - torch module object
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Enables gradient calculation for params of a given torch model

    Parametres:
        model - torch module object
    """
    for param in model.parameters():
        param.requires_grad = True

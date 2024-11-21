import torch
def freeze_model(model:torch.nn.Module):
    """Disables gradient calculation for params of a given torch model

    Args:
        model (torch.nn.Module): torch module object
    """    
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model:torch.nn.Module):
    """Enables gradient calculation for params of a given torch model

    Args:
        model (torch.nn.Module): torch module object
    """    
    
    for param in model.parameters():
        param.requires_grad = True

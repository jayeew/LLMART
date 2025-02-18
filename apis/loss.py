# coding: utf8
import torch
import torch.nn as nn

def loss_adv(loss_name, outputs, labels, target_labels, target, device):
    '''The function to create loss function.'''
    if loss_name=="ce":
        loss = nn.CrossEntropyLoss()
        
        if target:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

    elif loss_name =='cw':
        if target:
            one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
            cost = cost.sum()
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
            cost = cost.sum()
    return cost
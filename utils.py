import os
import json
import jittor as jt
import jittor.nn as nn
from jittor import Module

from datetime import datetime
import numpy as np


def cls_acc(output, target, topk=1):
    pred = output.topk(5, 1, True, True)[1].transpose() # int32 <class 'jittor.jittor_core.Var'> 
    correct = pred.equal(target.view(1, -1).expand_as(pred)) # bool <class 'jittor.jittor_core.Var'>
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) # <class 'float'>
    acc = 100 * acc / target.shape[0]
    return acc


def kl_loss(logit1, logit2):
    p = nn.log_softmax(logit1, dim=1)
    q = nn.softmax(logit2, dim=1) + 1e-8

    kl_div = nn.KLDivLoss(reduction='none')
    kl = kl_div(p, q).sum(dim=1)

    return kl.mean()


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def execute(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = nn.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * nn.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def save_model(model, save_path, test_acc):
    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d%H%M%S")
    mode_save_dict = {k: v for k, v in model.state_dict().items()} #clip_model模型不需要存储
    save_path = os.path.join(save_path, f'model-{formatted_time}.pkl')
    save_state = {
        'model': mode_save_dict, 
        'test_acc': test_acc
    }
    jt.save(save_state, save_path)  
    print(f"save to {save_path} !")

def load_model(model, model_path):
    print(model_path)
    checkpoint = jt.load(model_path)
    msg = model.load_state_dict(checkpoint['model']) # update
    test_acc = checkpoint['test_acc']
    print(f"=> loaded successfully '{model_path}', test_acc :{test_acc}")
    return model
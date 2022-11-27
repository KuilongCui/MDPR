# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn as nn
import math

__all__ = [
    "Linear",
    "ArcSoftmax",
    "CosSoftmax",
    "CircleSoftmax",
    "CurricularSoftmax"
]


class Linear(nn.Module):
    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits, targets):
        return logits.mul_(self.s)

    def extra_repr(self):
        return f"num_classes={self.num_classes}, scale={self.s}, margin={self.m}"


class CosSoftmax(Linear):
    r"""Implement of large margin cosine distance:
    """

    def forward(self, logits, targets):
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], self.m)
        logits[index] -= m_hot
        logits.mul_(self.s)
        return logits


class ArcSoftmax(Linear):

    def forward(self, logits, targets):
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], self.m)
        logits.acos_()
        logits[index] += m_hot
        logits.cos_().mul_(self.s)
        return logits


class CircleSoftmax(Linear):

    def forward(self, logits, targets):
        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        # When use model parallel, there are some targets not in class centers of local rank
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)

        # print(index)
        # print(m_hot.shape)
        # print(targets[index, None])
        # assert 1==0

        m_hot.scatter_(1, targets[index, None], 1)

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)

        logits[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)

        neg_index = torch.where(targets == -1)[0]
        logits[neg_index] = logits_n[neg_index]

        logits.mul_(self.s)

        return logits
    
class CurricularSoftmax(nn.Module):
    def __init__(self, num_classes, scale, margin):
        super(CurricularSoftmax, self).__init__()
        self.num_classes = num_classes

        m = margin
        self.m = margin
        self.s = scale
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # self.register_buffer('t', torch.zeros(1))

    def forward(self, cos_theta, label):
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))

        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m # cos(target+margin)
        
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        
        hard_example = cos_theta[mask]

        with torch.no_grad():
            self.t = target_logit.mean() * 0.001 + (1 - 0.001) * self.t
        cos_theta[mask] = hard_example * (self.t.half() + hard_example)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit.half())
        output = cos_theta * self.s
        
        return output

    def extra_repr(self):
        return f"num_classes={self.num_classes}, scale={self.s}, margin={self.m}"

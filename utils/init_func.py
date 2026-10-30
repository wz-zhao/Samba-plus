#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    """Include every trainable parameter once, including direct SSM parameters."""
    no_decay_ids = set()
    norm_types = (norm_layer, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.GroupNorm, nn.LayerNorm)
    for child in module.modules():
        if isinstance(child, norm_types):
            no_decay_ids.update(id(p) for p in child.parameters(recurse=False))
    decay, no_decay = [], []
    seen = {id(p) for group in weight_group for p in group['params']}
    for name, param in module.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        exclude = (id(param) in no_decay_ids or name.endswith('bias')
                   or getattr(param, '_no_weight_decay', False))
        (no_decay if exclude else decay).append(param)
    weight_group.extend([dict(params=decay, lr=lr),
                         dict(params=no_decay, weight_decay=0.0, lr=lr)])
    return weight_group

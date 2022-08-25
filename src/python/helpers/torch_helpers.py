from collections import OrderedDict
from contextlib import contextmanager
from time import time
from typing import Generator, List

import torch.cuda
from torch import nn


# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
def fix_data_parallel_state_dict(state_dict: dict) -> dict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


@contextmanager
def cuda_perf_clock(name: str, do_log: bool = True) -> Generator:
    start = time()
    try:
        yield
    finally:
        if do_log:
            torch.cuda.synchronize()
            end = time()
            print(f"{name} took {(end - start) * 1000.0} milliseconds.")


def initialize_linear(fc: nn.Linear) -> None:
    nn.init.normal_(fc.weight, 0, 0.01)
    nn.init.constant_(fc.bias, 0)


def initialize_conv2d(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def initialize_batch_norm2d(bn: nn.BatchNorm2d) -> None:
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)


def get_vgg_conv_blocks(vgg_model: nn.Module) -> List[nn.Module]:
    modules = []
    current_layer = []
    for module in vgg_model.features.modules():
        if isinstance(module, nn.Sequential):
            continue
        current_layer.append(module)
        if isinstance(module, nn.MaxPool2d):
            modules.append(nn.Sequential(*current_layer))
            current_layer = []
    return modules


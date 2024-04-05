import copy
from typing import Callable, Tuple

import torch
from thop.profile import profile  # using to get params and flops
from torch import nn


class ModelInspection:
    def __init__(self, verbose=False, input_shape=(224, 224, 3)) -> None:  # (H, W, C)
        self.verbose = verbose
        self.input_shape = input_shape

    def __call__(self, model, logger):
        # the number of parameters
        n_params = sum(param.numel() for param in model.parameters())
        # the number of gradients
        n_grads = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )

        if self.verbose:
            message = "___MODEL INFOMATION___\n"
            message += "\tModel Detail:\n"
            for i, (name, params) in enumerate(model.named_parameters()):
                name = name.replace("module_list.", "")
                message += f"\t  [...] layer: {i}, name: {name}, gradient: {params.requires_grad}, params: {params.numel()}, "
                message += f"shape: {list(params.shape)}, mu: {params.mean().item()}, sigma: {params.std().item()}\n"
        # get FLOPs
        try:
            device = next(model.parameters()).device
            dummy_image = torch.zeros(
                size=(1, self.input_shape[2], self.input_shape[0], self.input_shape[1]),
                device=device,
            )
            total_ops, total_params = profile(
                copy.deepcopy(model), inputs=(dummy_image,), verbose=False
            )  # MACs, params
            total_ops, total_params = round(total_ops / 1e9, 2), round(
                total_params / 1e6, 2
            )  # GMACs, Mparams
        except (ImportError, Exception):
            total_ops, total_params = "-", "-"

        message += "\tModel Summary:\n"
        message += f"\t  [...] Layers: {len(list(model.modules()))}, Parameters: {n_params}, Gradients: {n_grads}\n"
        message += f"\t  [...] Params (M): {total_params}, MACs (G): {total_ops}\n"

        logger.info(message)
        print(message)

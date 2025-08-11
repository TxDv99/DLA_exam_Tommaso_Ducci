from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from operator import mul
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


class MyResNet(nn.Module):
    def __init__(self, layers_tuple_list=None, data_shape=None, skip_dict=None, debug=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_projections = nn.ModuleDict()
        self.skips = skip_dict
        self._previous_is_conv = False
        self._hidden_size = 0
        self.data_shape = data_shape
        self.flatten = nn.Flatten()
        self.debug = debug
        self.device = None

        if layers_tuple_list is not None:
            for layer_number, layer_info in enumerate(layers_tuple_list):
                last_layer = layer_number == len(layers_tuple_list) - 1
                layer_type = layer_info[0]
                act_fun = layer_info[-1]

                if layer_type == "Linear":
                    self.addLinear(layer_info, act_fun, last_layer)
                elif layer_type == "Conv2d":
                    self.addConv(layer_info, act_fun, last_layer)
                elif layer_type in ["BatchNorm2d", "MaxPool2d", "Dropout"]:
                    layer_cls = getattr(nn, layer_type)
                    self.layers.append(layer_cls(layer_info[1]))
                else:
                    raise ValueError(f"Unexpected layer type: {layer_type}")

        # Setup skip projection layers
        if self.skips is not None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, *self.data_shape)
                outputs = []
                x = dummy_input
                for i, layer in enumerate(self.layers):
                    if layer.__class__.__name__ == "Linear" and len(x.shape) > 1:
                        x = x.flatten(start_dim=1)
                    x = layer(x)
                    outputs.append(x)

                for from_idx, to_idx in self.skips.items():
                    if from_idx >= len(outputs) or to_idx >= len(outputs):
                        raise IndexError("Invalid skip connection index.")
                    x_from = outputs[from_idx]
                    x_to = outputs[to_idx]
                    if x_from.shape != x_to.shape:
                        if x_from.ndim == 4:
                            proj = nn.Conv2d(
                                in_channels=x_from.shape[1],
                                out_channels=x_to.shape[1],
                                kernel_size=1,
                                stride=(
                                    max(1, x_from.shape[2] // x_to.shape[2]),
                                    max(1, x_from.shape[3] // x_to.shape[3])
                                )
                            )
                        elif x_from.ndim == 2:
                            proj = nn.Linear(x_from.shape[1], x_to.shape[1])
                        else:
                            raise ValueError("Unsupported projection for skip connection.")
                        self.skip_projections[f"{from_idx}->{to_idx}"] = proj

    def addConv(self, layer_info, act_fun, last_layer=False):
        padding_m = layer_info[6] if len(layer_info) >= 6 else 'zeros'
        conv = nn.Conv2d(
            in_channels=layer_info[1],
            out_channels=layer_info[2],
            kernel_size=layer_info[3],
            stride=layer_info[4],
            padding=layer_info[5],
            padding_mode=padding_m
        )
        self.layers.append(conv)
        self._previous_is_conv = True
        if not last_layer:
            self.layers.append(getattr(nn, act_fun)() if act_fun else nn.ReLU())

    def addLinear(self, layer_info, act_fun, last_layer=False):
        if self._previous_is_conv:
            n_channels, H_in, W_in = self.data_shape
            with torch.no_grad():
                dummy = torch.zeros(1, n_channels, H_in, W_in)
                out = nn.Sequential(*self.layers)(dummy)
                self._hidden_size = reduce(mul, tuple(out.shape))
            self._build_adapter(layer_info)
            self._previous_is_conv = False

        in_size = layer_info[1]
        width = layer_info[2]
        depth = layer_info[3]
        out_size = layer_info[4]

        dims = [in_size] + [width] * depth + [out_size]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if not last_layer:
                self.layers.append(getattr(nn, act_fun)() if act_fun else nn.ReLU())

    def _build_adapter(self, layer_info):
        if layer_info[1] != self._hidden_size:
            adapter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self._hidden_size, layer_info[1])
            )
            self.layers.append(adapter)

    
    def _reproject(self, x_from, from_idx, to_idx):
        key = f"{from_idx}->{to_idx}"
        if key in self.skip_projections:
            return self.skip_projections[key](x_from)
        else:
            return x_from


    def forward(self, x):
        skip_dict = self.skips
        skip_inputs = {}

        for index, layer in enumerate(self.layers):
            if layer.__class__.__name__ == "Linear" and len(x.shape) > 1:
                x = x.flatten(start_dim=1)

            x = layer(x)

            if skip_dict is not None:
                if index in skip_dict.keys():
                    idx_to = skip_dict[index]
                    skip_inputs[idx_to] = self._reproject(x, index, idx_to)
                elif index in skip_dict.values():
                    x = x + skip_inputs[index]

        return x

    def test(self, dataset, batch_size=64, num_workers=2, loss_fn=torch.nn.CrossEntropyLoss(), plot = False):
        device = next(self.parameters()).device
        self.eval()
        total_loss, correct, total = 0.0, 0, 0

       
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        y_pred = []
        y_true = []
        with torch.no_grad():
            for xs, ys in loader:
                xs, ys = xs.to(device), ys.to(device)
                logits = self(xs)
                loss = loss_fn(logits, ys)
                total_loss += loss.item() * xs.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == ys).sum().item()
                total += ys.size(0)
                
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(ys.cpu().tolist())

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Test set loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
        self.train()

        if plot:
            cm = confusion_matrix(np.array(y_true), np.array(y_pred))
            cmn = cm.astype(np.float32)
            cmn /= cmn.sum(1, keepdims=True)
            cmn = (100 * cmn).astype(np.int32)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            disp = ConfusionMatrixDisplay(cmn, display_labels=dataset.classes)
            disp.plot(ax=ax, cmap='viridis')
            
            # Ruota le etichette sull'asse X
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.show()

        return avg_loss, accuracy

    def get_submodel(self, up_to_layer):
        submodel = MyResNet()
        submodel.layers = nn.ModuleList(self.layers[:up_to_layer])
        submodel._previous_is_conv = self._previous_is_conv
        submodel.data_shape = self.data_shape
        submodel._hidden_size = self._hidden_size
        submodel.device = self.device
        return submodel

    def show(self):
        for layer in self.layers:
            print(layer)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

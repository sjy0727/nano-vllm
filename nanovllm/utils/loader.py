import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# def load_model(model: nn.Module, path: str):
#     packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
#     for file in glob(os.path.join(path, "*.safetensors")):
#         with safe_open(file, "pt", "cpu") as f:
#             for weight_name in f.keys():
#                 for k in packed_modules_mapping:
#                     if k in weight_name:
#                         v, shard_id = packed_modules_mapping[k]
#                         param_name = weight_name.replace(k, v)
#                         param = model.get_parameter(param_name)
#                         weight_loader = getattr(param, "weight_loader")
#                         weight_loader(param, f.get_tensor(weight_name), shard_id)
#                         break
#                 else:
#                     param = model.get_parameter(weight_name)
#                     weight_loader = getattr(param, "weight_loader", default_weight_loader)
#                     weight_loader(param, f.get_tensor(weight_name))


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 获取所有权重文件路径
    files = glob(os.path.join(path, "*.safetensors"))
    
    # 计算张量总数以初始化进度条
    total_tensors = 0
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            total_tensors += len(f.keys())

    if total_tensors == 0:
        # 如果找不到权重文件，提前告知用户
        print(f"Warning: No '.safetensors' files found in {path}. Model weights are not loaded.")
        return

    with tqdm(total=total_tensors, unit="tensors", desc=f"Loading weights from {os.path.basename(path).strip('/')}") as pbar:
        for file in files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    pbar.update(1)
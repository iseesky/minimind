import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=None):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        scaling = (rank if lora_alpha is None else lora_alpha) / rank
        self.register_buffer("scaling", torch.tensor(scaling, dtype=torch.float32))
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling


# [PATCH] 允许注入 LoRA 的投影层白名单（按模块名末段匹配）。
# 旧实现用 in_features == out_features 判据，只覆盖 q_proj/o_proj 这类方阵：
# GQA 下 k_proj/v_proj、整个 FFN（gate/up/down_proj）以及 MoE 专家全部漏注入。
# lm_head 与 embed_tokens 权重共享且词表维度巨大，MoE 的 gate 是路由层，均刻意排除。
# 兼容性：新注入层的 B 初始化为 0，旧 checkpoint 经 load_lora(strict=False) 加载后
# 缺失键保持 B=0，delta=0，输出与旧模型完全一致。
LORA_TARGET_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj')


def apply_lora(model, rank=16, lora_alpha=None, target_suffixes=LORA_TARGET_SUFFIXES):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split('.')[-1] in target_suffixes:
            lora = LoRA(module.in_features, module.out_features, rank=rank, lora_alpha=lora_alpha).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state, strict=False)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() if k.endswith('.weight') else v.cpu().float()
                          for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().float()
            if hasattr(module, 'lora'):
                delta = (module.lora.B.weight.data.float() @ module.lora.A.weight.data.float()) * module.lora.scaling.float()
                state_dict[f'{name}.weight'] += delta.cpu()
            state_dict[f'{name}.weight'] = state_dict[f'{name}.weight'].half()
    torch.save(state_dict, save_path)

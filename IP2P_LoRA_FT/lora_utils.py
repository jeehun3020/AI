import torch
import torch.nn as nn


class LoRALinearLayer(nn.Module):
    """
    LoRA Linear Layer - 저랭크 적응(Low-Rank Adaptation) 구현
    
    Args:
        in_features: 입력 차원
        out_features: 출력 차원
        rank: 저랭크 행렬의 랭크 (r)
        alpha: 스케일링 파라미터
    """
    def __init__(self, in_features, out_features, rank=8, alpha=8, device=None, dtype=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))
    
    def forward(self, x):
        """
        Forward pass: x @ lora_A @ lora_B * scaling
        
        Args:
            x: 입력 텐서
        Returns:
            LoRA 변환 후 출력 텐서
        """
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return lora_output


class LoRALayer(nn.Module):
    """
    베이스 레이어와 LoRA를 결합하는 래퍼 레이어
    
    Args:
        base_layer: 원본 nn.Linear 레이어
        rank: LoRA 랭크
        alpha: LoRA 알파 (스케일링 파라미터)
    """
    def __init__(self, base_layer, rank=8, alpha=8):
        super().__init__()
        self.base_layer = base_layer
        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype

        self.lora = LoRALinearLayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            device=device,
            dtype=dtype
        )
    
    def forward(self, x):
        """
        Forward pass: base_layer(x) + lora(x)
        
        Args:
            x: 입력 텐서
        Returns:
            베이스 레이어 출력 + LoRA 적응 출력
        """
        return self.base_layer(x) + self.lora(x)


def apply_lora_to_unet_full(unet, rank=8, alpha=8, target_modules=["to_q", "to_k", "to_v", "to_out.0"]):
    """
    전체 UNet의 모든 어텐션 레이어에 LoRA 적용
    
    Args:
        unet: UNet2DConditionModel
        rank: LoRA 랭크
        alpha: LoRA 알파
        target_modules: LoRA를 적용할 모듈 이름 리스트
    
    Returns:
        lora_layers: 적용된 LoRA 레이어 리스트
        trainable_params: 학습 가능한 파라미터 총 개수
    """
    lora_layers = []
    
    def apply_lora_recursive(module, name=""):
        """
        모델을 재귀적으로 순회하며 타겟 Linear 레이어에 LoRA 적용
        """
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear) and any(target in full_name for target in target_modules):
                lora_layer = LoRALayer(child_module, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                lora_layers.append(lora_layer)
            else:
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(unet)
    
    trainable_params = sum(p.numel() for lora in lora_layers for p in lora.lora.parameters())
    
    return lora_layers, trainable_params


def apply_lora_to_unet_selective(unet, rank=8, alpha=8, selected_blocks=["up", "mid"], 
                                  target_modules=["to_q", "to_k", "to_v", "to_out.0"]):
    """
    UNet의 선택된 블록에만 LoRA 적용
    
    Args:
        unet: UNet2DConditionModel
        rank: LoRA 랭크
        alpha: LoRA 알파
        selected_blocks: LoRA를 적용할 블록 타입 리스트 ["down", "mid", "up"]
        target_modules: LoRA를 적용할 모듈 이름 리스트
    
    Returns:
        lora_layers: 적용된 LoRA 레이어 리스트
        trainable_params: 학습 가능한 파라미터 총 개수
    """
    lora_layers = []
    
    def should_apply_lora(full_name):
        """
        블록 선택 기준에 따라 해당 레이어에 LoRA를 적용할지 결정
        """
        if "down_blocks" in full_name and "down" in selected_blocks:
            return True
        if "mid_block" in full_name and "mid" in selected_blocks:
            return True
        if "up_blocks" in full_name and "up" in selected_blocks:
            return True
        return False
    
    def apply_lora_recursive(module, name=""):
        """
        모델을 재귀적으로 순회하며 선택된 블록에 LoRA 적용
        """
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear) and any(target in full_name for target in target_modules):
                if should_apply_lora(full_name):
                    lora_layer = LoRALayer(child_module, rank=rank, alpha=alpha)
                    setattr(module, child_name, lora_layer)
                    lora_layers.append(lora_layer)
            else:
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(unet)
    
    trainable_params = sum(p.numel() for lora in lora_layers for p in lora.lora.parameters())
    
    return lora_layers, trainable_params
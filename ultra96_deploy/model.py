import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ReLU6(nn.Module):
    # min(max(0,x),6) — bounded for INT8-safe HLS export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0, max=6)


class MTLPickleballNet(nn.Module):
    # shared trunk → regression head (6D racket state) + classification head (6 shot types)
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        regression_output_dim: int = 6,
        num_classes: int = 6,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.regression_output_dim = regression_output_dim
        self.num_classes = num_classes
        
        # Build shared trunk
        trunk_layers = []
        current_dim = input_dim
        
        for i in range(num_hidden_layers):
            trunk_layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                trunk_layers.append(nn.BatchNorm1d(hidden_dim))
            trunk_layers.append(ReLU6())
            if dropout_rate > 0:
                trunk_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        self.shared_trunk = nn.Sequential(*trunk_layers)
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ReLU6(),
            nn.Linear(hidden_dim // 2, regression_output_dim)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ReLU6(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_trunk(x)
        regression_output = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)
        return regression_output, classification_logits
    
    def predict_shot_type(self, x: torch.Tensor) -> torch.Tensor:
        _, logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    # down-weights easy examples: FL(p) = -alpha * (1-p)^gamma * log(p)
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class MTLLoss(nn.Module):
    # weighted sum of MSE (regression) + CE/Focal (classification)
    
    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        self.mse_loss = nn.MSELoss()
        
        if use_focal_loss:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        regression_pred: torch.Tensor,
        regression_target: torch.Tensor,
        classification_logits: torch.Tensor,
        classification_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reg_loss = self.mse_loss(regression_pred, regression_target)
        cls_loss = self.ce_loss(classification_logits, classification_target)
        
        total_loss = (
            self.regression_weight * reg_loss + 
            self.classification_weight * cls_loss
        )
        
        return total_loss, reg_loss, cls_loss


def create_model(config: dict) -> MTLPickleballNet:
    # factory: creates model from config dict
    return MTLPickleballNet(
        input_dim=config.get('input_dim', 6),
        hidden_dim=config.get('hidden_dim', 64),
        num_hidden_layers=config.get('num_hidden_layers', 2),
        regression_output_dim=config.get('regression_output_dim', 6),
        num_classes=config.get('num_classes', 6),
        dropout_rate=config.get('dropout_rate', 0.2),
        use_batch_norm=config.get('use_batch_norm', True)
    )


def export_to_onnx(
    model: MTLPickleballNet,
    filepath: str,
    input_dim: int = 6,
    opset_version: int = 11
) -> None:
    # exports model to ONNX with dynamic batch axis
    model_cpu = model.cpu()
    model_cpu.eval()
    dummy_input = torch.randn(1, input_dim)
    # Use legacy ONNX export for compatibility
    torch.onnx.export(
        model_cpu,
        dummy_input,
        filepath,
        input_names=['ball_state'],
        output_names=['racket_state', 'shot_logits'],
        dynamic_axes={
            'ball_state': {0: 'batch_size'},
            'racket_state': {0: 'batch_size'},
            'shot_logits': {0: 'batch_size'}
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True
    )
    print(f"Model exported to ONNX: {filepath}")


if __name__ == '__main__':
    print("Testing MTLPickleballNet...")
    
    config = {
        'input_dim': 6,
        'hidden_dim': 64,
        'num_hidden_layers': 2,
        'regression_output_dim': 6,
        'num_classes': 6,
        'dropout_rate': 0.2,
        'use_batch_norm': True
    }
    
    model = create_model(config)
    print(f"Model created with {model.get_num_parameters()} parameters")
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 6)
    reg_out, cls_out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Regression output shape: {reg_out.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    # Test loss computation
    loss_fn = MTLLoss(regression_weight=1.0, classification_weight=1.0)
    reg_target = torch.randn(batch_size, 6)
    cls_target = torch.randint(0, 6, (batch_size,))
    
    total_loss, reg_loss, cls_loss = loss_fn(reg_out, reg_target, cls_out, cls_target)
    print(f"\nLoss values:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Regression (MSE): {reg_loss.item():.4f}")
    print(f"  Classification (CE): {cls_loss.item():.4f}")
    print("\nModel test passed!")

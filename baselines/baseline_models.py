"""
基线模型集合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Dict, Union


class VAEBaseline(nn.Module):
    """VAE基线模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 latent_dim: int = 64, n_classes: int = 50):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 分类器
        self.classifier = nn.Linear(latent_dim, n_classes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # 解码
        recon_x = self.decoder(z)

        # 分类
        logits = self.classifier(z)

        return {
            'recon_x': recon_x,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'logits': logits
        }

    def loss(self, outputs: Dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 重建损失
        recon_loss = F.mse_loss(outputs['recon_x'], x)

        # KL损失
        kl_loss = -0.5 * torch.mean(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        )

        # 分类损失
        class_loss = F.cross_entropy(outputs['logits'], y)

        return recon_loss + kl_loss + class_loss


class SimpleNN(nn.Module):
    """简单神经网络基线"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_classes: int = 50):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.features(x)
        logits = self.classifier(features)

        if return_features:
            return {'logits': logits, 'features': features}
        return logits

def get_sklearn_baselines(n_classes: int) -> Dict:
    """获取sklearn基线模型"""
    return {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42)
    }
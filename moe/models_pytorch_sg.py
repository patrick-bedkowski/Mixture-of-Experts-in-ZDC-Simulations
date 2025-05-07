import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Define the Router Network
class RouterNetwork(nn.Module):
    def __init__(self, cond_dim, num_generators):
        super(RouterNetwork, self).__init__()
        self.name = "router-architecture-2-gumbel-softmax"
        self.num_generators = num_generators
        self.fc_layers = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, self.num_generators)
        )

    def forward(self, cond, tau=1.0, hard=False):
        logits = self.fc_layers(cond)  # [B, E] raw scores
        gates = F.gumbel_softmax(logits, tau=tau, hard=hard)
        # gates now ∈ [0,1]⁽ᴮ⁾ˣᴱ, sums to 1 per batch element;
        # if hard=True, a straight-through one-hot approximation
        return gates, logits


class GeneratorUnified(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength, n_experts=3):
        self.name = "Generator-MultiOutput-v2"
        self.di_strength = di_strength
        self.in_strength = in_strength
        self.n_experts = n_experts

        super(GeneratorUnified, self).__init__()
        # Keep the exact same architecture as your original
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.expert_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256 * 20 * 10),
                nn.BatchNorm1d(256 * 20 * 10),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.1, inplace=True)
            )
            for _ in range(n_experts)
        ])

        self.upsample = nn.Upsample(scale_factor=(2, 2))
        # Vectorized conv layers using groups for expert isolation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256 * self.n_experts, 128 * self.n_experts, kernel_size=(3, 3), groups=n_experts),
            nn.BatchNorm2d(128 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(128 * self.n_experts, 64 * self.n_experts, kernel_size=(4, 4), groups=n_experts),
            nn.BatchNorm2d(64 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64 * self.n_experts, 32 * self.n_experts, kernel_size=(6, 4), groups=n_experts),
            nn.BatchNorm2d(32 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=(2, 1)),
            nn.Conv2d(32 * self.n_experts, 1 * self.n_experts, kernel_size=(5, 1)),
            nn.ReLU()
        )

    def forward(self, noise, cond, tau=1.0):
        # 1) shared projection
        x = torch.cat([noise, cond], dim=1)
        x = self.fc1(x)                   # [B, 256]

        # 2) sparse separate fc2 for each expert
        expert_outputs = [fc(x) for fc in self.expert_fcs]  # lista [B, 256*20*10]
        x = torch.cat(expert_outputs, dim=1)  # [B, (256*20*10) * n_experts]
        x = x.view(-1, 256 * self.n_experts, 20, 10)


        # 3) conv and unpack experts
        x = self.upsample(x)  # torch.Size([128, 256, 60, 20])
        x = self.conv_layers(x)         # [B, n_experts, H, W]
        x = x.unsqueeze(2)  # Shortcut to [B, n_experts, 1, H, W]
        return x


class Discriminator(nn.Module):
    def __init__(self, cond_dim):
        super(Discriminator, self).__init__()
        self.name = "Discriminator-3-expert-features"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 12 * 12 + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        out = self.sigmoid(out)
        return out, latent


class AuxRegUnified(nn.Module):
    def __init__(self, n_experts):
        super(AuxRegUnified, self).__init__()
        self.name = "aux-architecture-2-unified"
        self.n_experts = n_experts
        # Feature extraction layers
        self.conv3 = nn.Conv2d(1*self.n_experts, 32*self.n_experts, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32*self.n_experts)
        self.leaky3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32*self.n_experts, 64*self.n_experts, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64*self.n_experts)
        self.leaky4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d((2, 1))

        self.conv5 = nn.Conv2d(64*self.n_experts, 128*self.n_experts, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128*self.n_experts)
        self.leaky5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d((2, 1))

        self.conv6 = nn.Conv2d(128*self.n_experts, 256*self.n_experts, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(256*self.n_experts)
        self.leaky6 = nn.LeakyReLU(0.1)

        # Dropout layers (separated from feature path)
        self.dropout = nn.Dropout(0.2)

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear((256 * 3 * 8)*self.n_experts, 2*self.n_experts)  # Update dimensions based on input size

    def forward(self, x):
        # Original forward pass with dropout
        x = self.pool3(self.dropout(self.leaky3(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout(self.leaky4(self.bn4(self.conv4(x)))))
        x = self.pool5(self.dropout(self.leaky5(self.bn5(self.conv5(x)))))
        x = self.dropout(self.leaky6(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        return self.dense(x)

    def get_features(self, img):
        x = self.pool3(self.leaky3(self.bn3(self.conv3(img))))
        x = self.pool4(self.leaky4(self.bn4(self.conv4(x))))
        x = self.pool5(self.leaky5(self.bn5(self.conv5(x))))
        x = self.leaky6(self.bn6(self.conv6(x)))
        features = x.mean([2, 3])  # Global average pooling
        return features  # [112, 256] E.g.

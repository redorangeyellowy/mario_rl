import torch
from torch import nn
import torch.nn.functional as F
import copy

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class MarioDuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2_value = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv2_adv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_value = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3_adv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.flat_value = nn.Flatten()
        self.flat_adv = nn.Flatten()
        self.fc1_value = nn.Linear(3136, 512)
        self.fc1_adv = nn.Linear(3136, 512)
        self.fc2_value = nn.Linear(512, output_dim)
        self.fc2_adv = nn.Linear(512, output_dim)
        
        online_module = []
        online_module.append(self.conv1)
        online_module.append(self.conv2_value)
        online_module.append(self.conv2_adv)
        online_module.append(self.conv3_value)
        online_module.append(self.conv3_adv)
        online_module.append(self.flat_value)
        online_module.append(self.flat_adv)
        online_module.append(self.fc1_value)
        online_module.append(self.fc1_adv)
        online_module.append(self.fc2_value)
        online_module.append(self.fc2_adv)
        
        self.online = nn.ModuleList(online_module)
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        
        if model == "online":
            
            x = F.relu(self.online[0](input))
            
            v = F.relu(self.online[1](x))
            v = F.relu(self.online[3](v))
            v = self.online[5](v)
            v = F.relu(self.online[7](v))
            v = self.online[9](v)
            
            a = F.relu(self.online[2](x))
            a = F.relu(self.online[4](a))
            a = self.online[6](a)
            a = F.relu(self.online[8](a))
            a = self.online[10](a)
            a_avg = torch.mean(a)
            
            q = v + a - a_avg
            return q

        elif model == "target":
            
            x = F.relu(self.target[0](input))
            
            v = F.relu(self.target[1](x))
            v = F.relu(self.target[3](v))
            v = self.target[5](v)
            v = F.relu(self.target[7](v))
            v = self.target[9](v)
            
            a = F.relu(self.target[2](x))
            a = F.relu(self.target[4](a))
            a = self.target[6](a)
            a = F.relu(self.target[8](a))
            a = self.target[10](a)
            a_avg = torch.mean(a)
            
            q = v + a - a_avg
            return q
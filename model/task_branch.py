import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskBranchAttributes(nn.Module):
    def __init__(self, in_nodes, hidden_nodes, out_nodes=8):
        super(TaskBranchAttributes, self).__init__()
        self.fc1 = nn.Linear(in_nodes, hidden_nodes)
        self.bn1 = nn.BatchNorm1d(hidden_nodes)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(hidden_nodes, out_nodes) ## 3 binary and 1 with 5 values also 8
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class TaskBranchLinearOutput(nn.Module):
    def __init__(self, in_nodes, hidden_nodes, out_nodes):
        super(TaskBranchLinearOutput, self).__init__()
        self.fc1 = nn.Linear(in_nodes, hidden_nodes)
        self.bn1 = nn.BatchNorm1d(hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, out_nodes) ## x y coords

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x

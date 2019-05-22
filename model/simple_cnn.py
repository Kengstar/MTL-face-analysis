import torch.nn as nn


class small_cnn(nn.Module):
    def __init__(self):
        super(small_cnn, self).__init__()           ##padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.out = nn.Linear(128 * 6 * 6, 10)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.max_pool(x)
        x = self.activation(self.conv3(x))
        x = self.max_pool(x)
        x = self.activation(self.conv4(x))
        x = x.view(-1, 128 * 6 * 6)
        x = self.softmax(self.out(x))
        return x

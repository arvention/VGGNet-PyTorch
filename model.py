import torch.nn as nn

"""
 different configurations of VGGNet
 max pooling layers are labeled as M
 other layers are all convolutional layers (ex: 3-64)
   the first number is the kernel size
   the second number is the output channels
 """

configs = {
    'A': ['3-64', 'M',
          '3-128', 'M',
          '3-256', '3-256', 'M',
          '3-512', '3-512', 'M',
          '3-512', '3-512', 'M'],

    'B': ['3-64', '3-64', 'M',
          '3-128', '3-128', 'M',
          '3-256', '3-256', 'M',
          '3-512', '3-512', 'M',
          '3-512', '3-512', 'M'],

    'C': ['3-64', '3-64', 'M',
          '3-128', '3-128', 'M',
          '3-256', '3-256', '1-256', 'M',
          '3-512', '3-512', '1-512', 'M',
          '3-512', '3-512', '1-512', 'M'],

    'D': ['3-64', '3-64', 'M',
          '3-128', '3-128', 'M',
          '3-256', '3-256', '3-256', 'M',
          '3-512', '3-512', '3-512', 'M',
          '3-512', '3-512', '3-512', 'M'],

    'E': ['3-64', '3-64', 'M',
          '3-128', '3-128', 'M',
          '3-256', '3-256', '3-256', '3-256', 'M',
          '3-512', '3-512', '3-512', '3-512', 'M',
          '3-512', '3-512', '3-512', '3-512', 'M']
}


class VGGNet(nn.Module):

    """VGGNet Architecture"""

    def __init__(self, config, batch_norm,
                 channels, class_count,
                 init_weights=True):

        super(VGGNet, self).__init__()
        self.config = config
        self.batch_norm = batch_norm
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

        if init_weights:
            self.init_weights()

    def get_conv_net(self):
        """
        returns the convolutional layers of the network
        """
        layers = []
        in_channels = self.channels

        for layer in configs[self.config]:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer = layer.split('-')
                kernel_size = int(layer[0])
                out_channels = int(layer[1])
                layers.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=1))

                if self.batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))

                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

        return nn.Sequential(*layers)

    def get_fc_net(self):
        """
        returns the fully connected layers of the network
        """
        layers = []

        layers.append(nn.Linear(512 * 7 * 7, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout())
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout())
        layers.append(nn.Linear(4096, self.class_count))

        return nn.Sequential(*layers)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, val=1)
                nn.init.constant_(module.bias, val=0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.constant_(module.bias, val=0)

    def forward(self, x):
        y = self.conv_net(x)
        y = y.view(-1, y.size(1) * y.size(2) * y.size(3))
        y = self.fc_net(y)
        return y

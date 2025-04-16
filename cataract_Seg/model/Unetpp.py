import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class NestedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_filters=32):
        super(NestedUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters

        # Downsampling path
        self.conv0_0 = ConvBlock(in_channels, n_filters)
        self.conv1_0 = ConvBlock(n_filters, n_filters * 2)
        self.conv2_0 = ConvBlock(n_filters * 2, n_filters * 4)
        self.conv3_0 = ConvBlock(n_filters * 4, n_filters * 8)
        self.conv4_0 = ConvBlock(n_filters * 8, n_filters * 16)

        # Upsampling path (Nested connections)
        self.conv0_1 = ConvBlock(n_filters + n_filters * 2, n_filters)
        self.conv1_1 = ConvBlock(n_filters * 2 + n_filters * 4, n_filters * 2)
        self.conv2_1 = ConvBlock(n_filters * 4 + n_filters * 8, n_filters * 4)
        self.conv3_1 = ConvBlock(n_filters * 8 + n_filters * 16, n_filters * 8)

        self.conv0_2 = ConvBlock(n_filters * 2 + n_filters, n_filters)
        self.conv1_2 = ConvBlock(n_filters * 4 + n_filters * 2, n_filters * 2)
        self.conv2_2 = ConvBlock(n_filters * 8 + n_filters * 4, n_filters * 4)

        self.conv0_3 = ConvBlock(n_filters * 3 + n_filters, n_filters)
        self.conv1_3 = ConvBlock(n_filters * 6 + n_filters * 2, n_filters * 2)

        self.conv0_4 = ConvBlock(n_filters * 4 + n_filters, n_filters)

        # Pooling and upsampling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.ModuleDict({
            'up1': nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=2, stride=2),
            'up2': nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2),
            'up3': nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2),
            'up4': nn.ConvTranspose2d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        })

        # Final output layer
        self.final = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Upsampling path with nested connections
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up['up1'](x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up['up2'](x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up['up3'](x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up['up4'](x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up['up1'](x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up['up2'](x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up['up3'](x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up['up1'](x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up['up2'](x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up['up1'](x1_3)], 1))

        # Final output layer
        output = self.final(x0_4)
        return output

# Example usage
if __name__ == "__main__":
    model = NestedUNet(in_channels=3, out_channels=1)
    x = torch.randn((1, 3, 256, 256))
    y = model(x)
    print(y.shape)
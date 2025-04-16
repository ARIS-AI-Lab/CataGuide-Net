import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class SegNet(nn.Module):
    def __init__(self, num_classes=16, num_landmarks=4):
        super(SegNet, self).__init__()
        self.num_landmarks = num_landmarks
        self.encoder_conv_00 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.encoder_conv_01 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.encoder_conv_10 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv_11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.encoder_conv_20 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv_21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.encoder_conv_22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.encoder_conv_30 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_conv_31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.encoder_conv_32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.encoder_conv_40 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.encoder_conv_41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.encoder_conv_42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Self-Attention layers
        self.attention_1 = SelfAttention(128)
        self.attention_2 = SelfAttention(256)
        self.attention_3 = SelfAttention(512)

        self.decoder_conv_42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder_conv_41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder_conv_40 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder_conv_32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder_conv_31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder_conv_30 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder_conv_22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_conv_21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_conv_20 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv_11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.decoder_conv_10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv_01 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_conv_00 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 上采样层
        self.upsample = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        '''
        self.landmark_fc = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, self.num_landmarks * 2)
        )

        self.landmark_class_fc = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_landmarks * (num_classes - 1))
        )
        '''

        self.num_classes_per_landmark = (num_classes - 1)

    def forward(self, x):
        x = F.relu(self.encoder_conv_00(x))
        x = F.relu(self.encoder_conv_01(x))
        x, idx1 = self.maxpool(x)

        x = F.relu(self.encoder_conv_10(x))
        x = F.relu(self.encoder_conv_11(x))
        # x = self.attention_1(x)
        x, idx2 = self.maxpool(x)

        x = F.relu(self.encoder_conv_20(x))
        x = F.relu(self.encoder_conv_21(x))
        x = F.relu(self.encoder_conv_22(x))
        # x = self.attention_2(x)
        x, idx3 = self.maxpool(x)

        x = F.relu(self.encoder_conv_30(x))
        x = F.relu(self.encoder_conv_31(x))
        x = F.relu(self.encoder_conv_32(x))
        x, idx4 = self.maxpool(x)

        x = F.relu(self.encoder_conv_40(x))
        x = F.relu(self.encoder_conv_41(x))
        x = F.relu(self.encoder_conv_42(x))
        x = self.attention_3(x)
        x, idx5 = self.maxpool(x)

        seg_output = self.unpool(x, idx5)
        seg_output = F.relu(self.decoder_conv_42(seg_output))
        seg_output = F.relu(self.decoder_conv_41(seg_output))
        seg_output = F.relu(self.decoder_conv_40(seg_output))

        seg_output = self.unpool(seg_output, idx4)
        seg_output = F.relu(self.decoder_conv_32(seg_output))
        seg_output = F.relu(self.decoder_conv_31(seg_output))
        seg_output = F.relu(self.decoder_conv_30(seg_output))

        seg_output = self.unpool(seg_output, idx3)
        seg_output = F.relu(self.decoder_conv_22(seg_output))
        seg_output = F.relu(self.decoder_conv_21(seg_output))
        seg_output = F.relu(self.decoder_conv_20(seg_output))

        seg_output = self.unpool(seg_output, idx2)
        seg_output = F.relu(self.decoder_conv_11(seg_output))
        seg_output = F.relu(self.decoder_conv_10(seg_output))

        seg_output = self.unpool(seg_output, idx1)
        seg_output = F.relu(self.decoder_conv_01(seg_output))
        seg_output = self.decoder_conv_00(seg_output)

        # seg_output = self.upsample(seg_output)

        # landmark_output = self.landmark_fc(x).view(-1, self.num_landmarks, 2)
        # landmark_class_output = self.landmark_class_fc(x).view(-1, self.num_landmarks, self.num_classes_per_landmark)

        # return seg_output, landmark_output, landmark_class_output
        return seg_output

if __name__ == "__main__":
    model = SegNet(num_classes=16, num_landmarks=4)
    input_tensor = torch.randn(3, 3, 512, 512)
    seg_output, landmark_output, landmark_class_output = model(input_tensor)
    print("Segmentation output shape:", seg_output.shape)  # Expected: (3, 16, 512, 512)
    print("Landmark output shape:", landmark_output.shape)  # Expected: (3, 4, 2)
    print("Landmark class output shape:", landmark_class_output.shape)  # Expected: (3, 4, 15)

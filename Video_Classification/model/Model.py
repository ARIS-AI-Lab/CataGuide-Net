import torch
import torch.utils.data
from torchsummary import summary
from torch.nn import Module, Dropout, BatchNorm1d, LeakyReLU, Linear, LogSoftmax, Sigmoid

# import torchviz
from model import timeception_pytorch
from core import config, utils


class Model(Module):
    """
    Define Timeception classifier.
    """

    def __init__(self):
        super(Model, self).__init__()

        # some configurations for the model
        n_tc_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS
        backbone_name = config.cfg.MODEL.BACKBONE_CNN
        feature_name = config.cfg.MODEL.BACKBONE_FEATURE
        n_tc_layers = config.cfg.MODEL.N_TC_LAYERS
        n_classes = config.cfg.MODEL.N_CLASSES
        is_dilated = config.cfg.MODEL.MULTISCALE_TYPE
        OutputActivation = Sigmoid if config.cfg.MODEL.CLASSIFICATION_TYPE == 'ml' else LogSoftmax
        n_channels_in, channel_h, channel_w = utils.get_model_feat_maps_info(backbone_name, feature_name)
        # print(n_channels_in, "*"*20)
        n_groups = int(n_channels_in / 128.0)

        input_shape = (None, n_channels_in, n_tc_timesteps, channel_h, channel_w)  # (C, T, H, W)
        self._input_shape = input_shape

        # define 4 layers of timeception
        self.timeception = timeception_pytorch.Timeception(input_shape, n_tc_layers, n_groups, is_dilated)  # (C, T, H, W)

        # get number of output channels after timeception
        n_channels_in = self.timeception.n_channels_out

        # define layers for classifier
        self.do1 = Dropout(0.5)
        self.l1 = Linear(n_channels_in, 512)
        self.bn1 = BatchNorm1d(512)
        self.ac1 = LeakyReLU(0.2)
        self.do2 = Dropout(0.25)
        self.l2 = Linear(512, n_classes)
        self.ac2 = OutputActivation()

    def forward(self, input):
        # feedforward the input to the timeception layers
        tensor = self.timeception(input)

        # max-pool over space-time
        bn, c, t, h, w = tensor.size()
        tensor = tensor.view(bn, c, t * h * w)
        tensor = torch.max(tensor, dim=2, keepdim=False)
        tensor = tensor[0]

        # dense layers for classification
        tensor = self.do1(tensor)
        tensor = self.l1(tensor)
        # print(tensor.shape)
        tensor = self.bn1(tensor)
        tensor = self.ac1(tensor)
        tensor = self.do2(tensor)
        tensor = self.l2(tensor)
        tensor = self.ac2(tensor)

        return tensor


if __name__ == '__main__':
    model = Model()
    input_tensor = torch.randn(4, 64, 270, 360, 3)
    # seg_output, landmark_output, landmark_class_output, edge_output = model(input_tensor)
    # output = model(input_tensor)
    # print(output.shape)
    summary(model, input_size=(128, 224, 224, 3), device='cpu')
from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet
import math
import os
import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.logits(x)
        return x


class my_2d1d(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.attention = attention
        self.dropout = dropout

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)
            state_dict = torch.load(path, map_location='cpu')
            spatial.load_state_dict(state_dict)
        elif 'eeg_image' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

            if fold is not None:
                path = os.path.join(self.root_dir, self.backbone_state_dict + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        else:
            raise ValueError("Unsupported modality!")

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size, attention=self.attention,
            dropout=self.dropout)
        self.regressor = nn.Linear(self.embedding_dim // 4, self.output_dim)
        # self.regressor = Sequential(
        #     BatchNorm1d(self.embedding_dim // 4),
        #     Dropout(0.4),
        #     Linear(self.embedding_dim // 4, self.output_dim))

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


class my_2d1ddy(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir='', input_dim_other=[128, 39]):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.attention = attention
        self.dropout = dropout
        self.other_feature_dim = input_dim_other  # input dimension of other modalities

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)
        state_dict = torch.load(path, map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size, attention=self.attention,
            dropout=self.dropout)

        self.temporal1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal2 = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        '''
        self.temporal3 = TemporalConvNet(
            num_inputs=self.other_feature_dim[2], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal4 = TemporalConvNet(
            num_inputs=self.other_feature_dim[3], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        '''

        self.encoder1 = nn.Linear(self.embedding_dim // 4, 32)
        self.encoder2 = nn.Linear(self.other_feature_dim[0] // 4, 32)
        self.encoder3 = nn.Linear(32, 32)

        self.encoderQ1 = nn.Linear(self.embedding_dim // 4, 32)
        self.encoderQ2 = nn.Linear(self.other_feature_dim[0] // 4, 32)
        self.encoderQ3 = nn.Linear(32, 32)

        self.encoderV1 = nn.Linear(self.embedding_dim // 4, 32)
        self.encoderV2 = nn.Linear(self.other_feature_dim[0] // 4, 32)
        self.encoderV3 = nn.Linear(32, 32)
        #self.encoder4 = nn.Linear(self.other_feature_dim[3] // 4, 32)
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 32)

        self.ln = nn.LayerNorm([3, 32])

        self.regressor = nn.Linear(224, self.output_dim)
        # self.regressor = Sequential(
        #     BatchNorm1d(self.embedding_dim // 4),
        #     Dropout(0.4),
        #     Linear(self.embedding_dim // 4, self.output_dim))

    def forward(self, x, x1, x2):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)

        if len(x1) > 1 and len(x2) > 1:
            x1 = x1.squeeze().transpose(1, 2).contiguous().float()
            x2 = x2.squeeze().transpose(1, 2).contiguous().float()
        else:
            x1 = x1.squeeze()[None, :, :].transpose(1, 2).contiguous().float()
            x2 = x2.squeeze()[None, :, :].transpose(1, 2).contiguous().float()
        #x3 = x3.transpose(1, 2).contiguous().float()
        #x4 = x4.transpose(1, 2).contiguous().float()

        x1 = self.temporal1(x1).transpose(1, 2).contiguous()
        x2 = self.temporal2(x2).transpose(1, 2).contiguous()
        #x3 = self.temporal3(x3).transpose(1, 2).contiguous()
        #x4 = self.temporal4(x4).transpose(1, 2).contiguous()

        x0 = self.encoder1(x)
        x1 = self.encoder2(x1.contiguous().view(num_batches * length, -1))
        x2 = self.encoder3(x2.contiguous().view(num_batches * length, -1))

        xq0 = self.encoderQ1(x)
        xq1 = self.encoderQ2(x1.contiguous().view(num_batches * length, -1))
        xq2 = self.encoderQ3(x2.contiguous().view(num_batches * length, -1))

        xv0 = self.encoderV1(x)
        xv1 = self.encoderV2(x1.contiguous().view(num_batches * length, -1))
        xv2 = self.encoderV3(x2.contiguous().view(num_batches * length, -1))
        #x3 = x3.contiguous().view(num_batches * length, -1)
        #x4 = x4.contiguous().view(num_batches * length, -1)

        x_K = torch.stack((x0, x1, x2), dim=-2)
        x_Q = torch.stack((xq0, xq1, xq2), dim=-2)
        x_V = torch.stack((xv0, xv1, xv2), dim=-2)

        x_QT = x_Q.permute(0, 2, 1)

        scores = torch.matmul(x_K, x_QT) / math.sqrt(32)

        scores = nn.functional.softmax(scores, dim=-1)

        out = torch.matmul(scores, x_V)

        out = self.ln(out + x_V)

        out = out.view(out.size()[0], -1)

        x = torch.cat((x, out), dim=-1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


class my_2dlstm(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality='frame', embedding_dim=512, hidden_dim=256,
                 output_dim=1, dropout=0.5, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        elif 'eeg_image' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

            if fold is not None:
                path = os.path.join(self.root_dir, self.backbone_state_dict + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        else:
            raise ValueError("Unsupported modality!")

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=self.dropout)
        self.regressor = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).contiguous()
        x, _ = self.temporal(x)
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


class my_temporal(nn.Module):
    def __init__(self, model_name, num_inputs=192, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5, cnn1d_dropout_rate=0.1,
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, bidirectional=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        if "1d" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                       kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim
            if bidirectional:
                input_dim = hidden_dim * 2

            self.regressor = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        features = {}
        if "lstm_only" in self.model_name:
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
            x = self.temporal(x).transpose(1, 2).contiguous()
        batch, time_step, temporal_feature_dim = x.shape

        x = x.view(-1, temporal_feature_dim)
        x = self.regressor(x).contiguous()
        x = x.view(batch, time_step, self.output_dim)
        return x
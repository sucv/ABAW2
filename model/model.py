from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet

import os
import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

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
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None,
                 attention=0, output_dim=1, kernel_size=5, adversarial=0, dropout=0.1, root_dir='', mlt_atten=0,
                 pred_AU=12, pred_Expre=12, au_class_n=12, expre_class_n=7):
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
        self.adversarial = adversarial
        self.mlt_atten = mlt_atten
        self.pred_AU = pred_AU
        self.pred_Expre = pred_Expre
        self.n_tasks = 0
        if output_dim:
            self.n_tasks += 1
            self.VA_index = self.n_tasks
        if self.pred_AU:
            self.n_tasks += 1
            self.AU_index = self.n_tasks
        if self.pred_Expre:
            self.n_tasks += 1
            self.Expre_index = self.n_tasks
        self.au_class_n = au_class_n
        self.expre_class_n = expre_class_n
        assert self.n_tasks > 0

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
        if self.output_dim:
            self.regressor = nn.Linear(self.embedding_dim // 4, self.output_dim)

        if self.adversarial:
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.embedding_dim // 16))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(self.embedding_dim // 16, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        if self.mlt_atten:
            from model.mlt_attention import Attn_Net_Gated
            self.mlt_atten_net = nn.Sequential(*[
                Attn_Net_Gated(indim=self.embedding_dim // 4, hiddim=self.embedding_dim // 4, n_tasks=self.n_tasks),
                nn.Softmax(dim=1)
            ])

        if self.pred_AU:
            self.auPred = nn.Sequential(*[
                nn.Linear(self.embedding_dim // 4, self.au_class_n),
                nn.Sigmoid()
            ])
        if self.pred_Expre:
            self.exprePred = nn.Sequential(*[
                nn.Linear(self.embedding_dim // 4, self.expre_class_n),
                # nn.Softmax(dim=1)
            ])

        # self.regressor = Sequential(
        #     BatchNorm1d(self.embedding_dim // 4),
        #     Dropout(0.4),
        #     Linear(self.embedding_dim // 4, self.output_dim))

    def forward(self, x, alpha=0):
        output = {}
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)
        x_AU = x
        x_Expre = x

        if self.mlt_atten:
            A = self.mlt_atten_net(x) #  n_tasks X num_batches*length
            X = [A[:, i].repeat(x.shape[-1], 1).transpose(1, 0) * x for i in range(self.n_tasks)]
            if self.output_dim:
                x = X[self.VA_index]
            elif self.pred_AU:
                x_AU = X[self.AU_index]
            elif self.pred_Expre:
                x_Expre = X[self.Expre_index]

        if self.adversarial:
            reverse_feature = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            output['domain_output'] = domain_output
        if self.output_dim:
            output['x'] = self.regressor(x).view(num_batches, length, -1)
        if self.pred_AU:
            output['AU'] = self.auPred(x_AU).view(num_batches, length, -1)
        if self.pred_Expre:
            output['Expre'] = self.exprePred(x_Expre).view(num_batches, length, -1)

        return output


class my_2dlstm(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality='frame', embedding_dim=512, hidden_dim=256,
                 output_dim=1, adversarial=0, bidirectional=1, dropout=0.5, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.adversarial = adversarial

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

        input_dim = self.hidden_dim
        if self.bidirectional:
            input_dim = self.hidden_dim * 2
        self.regressor = nn.Linear(input_dim, self.output_dim)

        if self.adversarial:
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(input_dim, input_dim // 4))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(input_dim // 4))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(input_dim // 4, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha=1):
        output = {}
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).contiguous()
        x, _ = self.temporal(x)
        x = x.contiguous().view(num_batches * length, -1)

        if self.adversarial:
            reverse_feature = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            output['domain_output'] = domain_output

        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        output['x'] = x

        return x


class my_temporal(nn.Module):
    def __init__(self, model_name, num_inputs=192, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5, cnn1d_dropout_rate=0.1,
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, adversarial=0, bidirectional=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        if "1d" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                       kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)
            ad_input_dim = cnn1d_channels[-1]
        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim
            if bidirectional:
                input_dim = hidden_dim * 2
            ad_input_dim = input_dim

            self.regressor = nn.Linear(input_dim, output_dim)

        if self.adversarial:
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(ad_input_dim, ad_input_dim // 4))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(ad_input_dim // 4, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha=1):
        output = {}
        features = {}
        if "lstm_only" in self.model_name:
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
            x = self.temporal(x).transpose(1, 2).contiguous()
        batch, time_step, temporal_feature_dim = x.shape

        x = x.view(-1, temporal_feature_dim)

        if self.adversarial:
            reverse_feature = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            output['domain_output'] = domain_output

        x = self.regressor(x).contiguous()
        x = x.view(batch, time_step, self.output_dim)
        output['x'] = x
        return x
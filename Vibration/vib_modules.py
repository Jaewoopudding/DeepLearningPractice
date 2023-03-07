import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft


class CNNModule(nn.Module):
    def __init__(self, sequence_shape, kernel_size=10, base_channel=64, module_count=2, pooling_coef=2,
                 cnn_padding='same', padding_mode='zeros', channel_multiplier=2, act_fn='relu'):
        assert pooling_coef * base_channel == int(pooling_coef * base_channel)
        act_func_dict = {
            'relu': nn.ReLU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'elu': nn.ELU(inplace=True),
        }
        super().__init__()
        self.sequence_shape = sequence_shape  # (channel, length) tuple
        self.kernel_size = kernel_size
        self.base_channel = base_channel
        self.module_count = module_count
        self.pooling_coef = pooling_coef
        self.padding_mode = padding_mode
        self.cnn_padding = cnn_padding
        self.channel_multiplier = channel_multiplier
        self.act_fn = act_func_dict[act_fn]

        self.model = self.build()

    def get_last_layer_length(self):
        return int(self.sequence_shape[-1] / (2 * self.module_count))

    def build(self):
        channel = self.base_channel
        length = self.sequence_shape[1]
        model = [nn.Conv1d(in_channels=self.sequence_shape[0], out_channels=channel, kernel_size=self.kernel_size,
                           padding=self.cnn_padding, padding_mode=self.padding_mode),
                 self.act_fn]

        priv_channel = channel
        channel *= self.channel_multiplier

        for _ in range(self.module_count):
            model += [
                nn.Conv1d(in_channels=priv_channel, out_channels=channel, kernel_size=self.kernel_size,
                          padding='same', padding_mode=self.padding_mode),
                self.act_fn,
                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=self.kernel_size,
                          padding='same', padding_mode=self.padding_mode),
                self.act_fn,
                nn.BatchNorm1d(channel),
                nn.MaxPool1d(2, stride=2)
            ]

            length = int(length / 2)
            priv_channel = channel
            channel *= self.channel_multiplier

        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class StatAnalModule(nn.Module):
    def __init__(self, length, complexity=2):
        super().__init__()
        self.length = length
        self.complexity = complexity

    @staticmethod
    def moment(x, order):
        return ((x - torch.mean(x, -1)[:, :, None]) ** order).sum(-1)

    def return_shape(self):
        if self.complexity == 2:
            features = 10
        elif self.complexity == 1:
            features = 7
        else:
            features = 5
        return features

    def forward(self, x):
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), -1))
        maxima = torch.max(x, -1).values
        minima = torch.min(x, -1).values
        result = torch.cat([torch.mean(x, -1), maxima, minima, rms, torch.var(x, -1)], -1)
        if self.complexity >= 1:
            skew = self.moment(x, 3) / (torch.std(x, -1)) ** 3 * self.length / (self.length - 1) / (self.length - 2)
            kurto = self.length / (self.length - 1) ** 2 * self.moment(x, 4) / (torch.std(x, -1)) ** 4 - 3
            result = torch.cat([result, skew, kurto], -1)

        if self.complexity >= 2:

            crest_factor = torch.abs(x).max(-1)[0] / rms
            meanabs = torch.abs(x).mean(-1)
            impulse_factor = (maxima - minima) / meanabs
            shape_factor = rms / meanabs
            result = torch.cat([result, crest_factor, impulse_factor, shape_factor], -1)

        return result


class CWTModule(nn.Module):
    def __init__(self, length, in_channel=1, cwt_filters=4):
        super().__init__()
        self.T = length  # 시퀀스 길이
        self.cwt_filters = cwt_filters  # not supported yet
        self.t_list = [int(self.T / self.cwt_filters) * i for i in range(1, cwt_filters + 1)]
        self.in_channel = in_channel

    def morlet(self, t_max, f0=6):
        t = torch.linspace(-2 * torch.pi, 2 * torch.pi, t_max)
        filt = torch.pi ** (1 / 4) * torch.exp(1j * f0 * t - (t ** 2) / 2).reshape(1, 1, 1, -1).real
        return torch.cat([filt for _ in range(self.in_channel)], axis=-2)

    def forward(self, x):
        x = x[:, None, :, :]
        return torch.squeeze(torch.cat([F.conv2d(x.float(), self.morlet(T).float(), padding='same') for T in self.t_list], axis=-2), 1)


class FFTModule(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, x):
        result = fft(x * torch.hamming_window(self.length)) / self.length
        return torch.cat([result.real, result.imag], axis=-2)


class MLPClassifier(nn.Module):
    def __init__(self, in_features, num_of_neurons, num_of_class):
        super().__init__()
        self.num_of_neurons = num_of_neurons
        self.num_of_class = num_of_class
        self.Linear1 = torch.nn.Linear(in_features=in_features, out_features=num_of_neurons)
        self.Linear2 = torch.nn.Linear(in_features=num_of_neurons, out_features=int(num_of_neurons/2))
        self.classifier = torch.nn.Linear(in_features=int(num_of_neurons/2), out_features=num_of_class)
        self.act_fn = torch.nn.ReLU()

    def forward(self, x):
        x = self.act_fn(self.Linear1(x))
        x = self.act_fn(self.Linear2(x))
        return self.classifier(x)


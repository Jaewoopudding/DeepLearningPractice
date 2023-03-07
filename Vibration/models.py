from vib_modules import *


class ConvolutionalLSTM(nn.Module):
    def __init__(self, shape, kernel_size=10, base_channel=64, module_count=2, pooling_coef=2, cnn_padding='same',
                 padding_mode='zeros', channel_multiplier=2, act_fn='relu',
                 lstm_hidden_size=64, lstm_num_layers=2, mlp_neurons=64,
                 stats_complexity=2, cwt_filters=4, num_of_class=5):
        super().__init__()
        shape = shape
        complexity_features = {0: 5, 1: 7, 2: 10}
        sequence_depth = shape[0] * (1 + 2 + cwt_filters)  # own signal + fft(r,i) + cwt filters

        self.statnet = StatAnalModule(shape[-1], stats_complexity)
        self.fft = FFTModule(shape[-1])
        self.cwt = CWTModule(shape[-1], cwt_filters=cwt_filters)
        self.cnn = CNNModule(sequence_shape=(sequence_depth, shape[-1]),
                             kernel_size=kernel_size,
                             base_channel=base_channel,
                             module_count=module_count,
                             pooling_coef=pooling_coef,
                             cnn_padding=cnn_padding,
                             padding_mode=padding_mode,
                             channel_multiplier=channel_multiplier,
                             act_fn=act_fn)
        self.lstm = nn.LSTM(input_size=base_channel * module_count * 2,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.classifier = MLPClassifier(
            in_features=int(lstm_hidden_size / 2) + shape[0] * complexity_features[stats_complexity],
            num_of_neurons=mlp_neurons, num_of_class=num_of_class)
        self.pooling = torch.nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        stat = self.statnet(x)
        x = torch.cat([x, self.fft(x), self.cwt(x)], dim=-2)
        x = torch.transpose(self.cnn(x), 2, 1)
        x = self.lstm(x)[0][:, -1, :]
        x = self.pooling(x)
        x = torch.cat([x, stat], -1)
        x = self.classifier(x)
        return x

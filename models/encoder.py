import torch

class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Convolution, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)

class Encoder(torch.nn.Module):
    def __init__(self, n_convolutions, embedding_dim, kernel_size, dropout_p=0.0):
        super(Encoder, self).__init__()

        convolutions = []

        for i in range(n_convolutions):
            conv_layer = torch.nn.Sequential(
                Convolution(embedding_dim, embedding_dim, kernel_size= kernel_size, stride=1, padding=int((kernel_size - 1) / 2, dilation=1, w_init_gain='relu')),
                torch.nn.BatchNorm1d(embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_p)
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(embedding_dim, int(embedding_dim / 2), 1, batch_first=True, bidirectional=True)


    def forward(self, data):
        for convolution in self.convolutions:
            data = convolution(data)

        data.transpose(1, 2)
        #TODO: probably need to add some logic to handle padding and input size
        outputs, hidden = self.lstm(data)

        return outputs




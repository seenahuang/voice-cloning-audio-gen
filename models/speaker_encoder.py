import torch


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(SpeakerEncoder, self).__init__()
        self.LSTM = torch.nn.LSTM(input_size, hidden_size, num_layers)
        #TODO: linear layer

    def forward(self, data):
        # TODO: get LSTM output, add in final linear layer to 256 dimensions, perform L2 normalization
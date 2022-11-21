import torch


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=3, embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.LSTM = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, embedding_size)

        self.device = device

    def forward(self, data):
        # TODO: get LSTM output, add in final linear layer to 256 dimensions, perform L2 normalization
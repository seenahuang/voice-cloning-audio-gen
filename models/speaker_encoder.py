import torch


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=3, embedding_size=256):
        """
        Initializes SpeakerEncoder model based on
        GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION by Wan et al. https://arxiv.org/pdf/1710.10467.pdf

        :param input_size: number of features for spectrogram inputs
        :param hidden_size: number hidden state features
        :param device: cpu or cuda if available
        :param num_layers: number of LSTM layers
        :param embedding_size: final speaker embedding size
        """
        super(SpeakerEncoder, self).__init__()
        self.LSTM = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, embedding_size)
        self.relu = torch.nn.ReLU()
        self.to(device)

    def forward(self, data):
        """
        Forward call of speaker encoder model

        :param data: Tensor of mel spectrograms in the shape of (batch, frames, channels)
        :return speaker embeddings of shape (batch, embedding_size)
        """
        _, (hidden, _) = self.LSTM(data)
        # apply linear to hidden state of the last layer
        embeddings = self.relu(self.linear(hidden[-1]))
        return embeddings / torch.linalg.norm(embeddings, dim=1, ord=2)

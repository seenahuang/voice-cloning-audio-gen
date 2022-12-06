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
        self.LSTM = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, embedding_size)
        self.relu = torch.nn.ReLU()
        self.to(device)

    def forward(self, data):
        """
        Forward call of speaker encoder model on multiple speakers.
        Each batch will have the same number of utterances for each speaker.

        :param data: Tensor of mel spectrograms in the shape of (num_speakers, num_utterances, frames*channels)
        :return speaker embeddings per utterance of shape (num_speakers, num_utterances, embedding_size)
        """
        out = None
        for speaker_data in data:
            _, (hidden, _) = self.LSTM(speaker_data)
            # apply linear to hidden state of the last layer
            embeddings = self.relu(self.linear(hidden[-1]))
            norm = torch.linalg.norm(embeddings, dim=1, ord=2)
            embeddings_norm = torch.div(torch.transpose(embeddings, 0, 1), norm)
            speaker_out = torch.transpose(embeddings_norm, 0, 1).unsqueeze(0)
            if out is None:
                out = speaker_out
            else:
                out = torch.cat((out, speaker_out), dim=0)
        return out

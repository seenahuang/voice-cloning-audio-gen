import utils
from models.speaker_encoder import SpeakerEncoder
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_librispeech = utils.load_train()
    # valid_librispeech = utils.load_validation()
    # test_librispeech = utils.load_test()
    train_loader = DataLoader(train_librispeech, batch_size=params['batch_size'], shuffle=True,
                              collate_fn=lambda x: utils.generate_spectrograms(x, device))

    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])

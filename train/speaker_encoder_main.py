import utils
from models.speaker_encoder import SpeakerEncoder
import torch


if __name__ == "__main__":
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")

    train_librispeech = utils.load_train()
    valid_librispeech = utils.load_validation()
    test_librispeech = utils.load_test()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])

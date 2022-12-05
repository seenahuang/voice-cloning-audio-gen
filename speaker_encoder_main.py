import utils
from models.speaker_encoder import SpeakerEncoder
from models.EndToEndLoss import EndToEndLoss
import torch
from torch.utils.data import DataLoader
import os

REMOVED_SPEAKERS = {1992, 8014, 7312, 445, 1183} # speaker IDs that have fewer than 80 utterances


if __name__ == "__main__":
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_librispeech = utils.load_train()
    # valid_librispeech = utils.load_validation()
    # test_librispeech = utils.load_test()

    if os.path.isfile('./data/processed/train.pt'):
        train_data = torch.load('./data/processed/train.pt')
    else:
        train_data = utils.preprocess_data(train_librispeech,
                                           REMOVED_SPEAKERS,
                                           params['num_speakers'],
                                           params['num_utterances'],
                                           params['waveform_length'],
                                           'train.pt')

    spec_size = (params['spec_length'], params['spec_channels'])
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                device,
                                                                                                spec_size))

    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])
    optimizer = torch.optim.SGD(encoder.parameters(), params['learning_rate'])
    #TODO: initialize E2E loss
    criterion = EndToEndLoss(10.0, -5.0, device)
    for epoch in range(params['epochs']):
        utils.train(epoch, train_loader, encoder, optimizer, criterion)

import utils
from models.speaker_encoder import SpeakerEncoder
from models.EndToEndLoss import EndToEndLoss
import torch
from torch.utils.data import DataLoader
import os
import sys

REMOVED_SPEAKERS = {1992, 8014, 7312, 445, 1183} # speaker IDs that have fewer than 80 utterances


def train(epoch, data_loader, model, optimizer, criterion):
    losses = torch.zeros(len(data_loader))

    for idx, spectrograms in enumerate(data_loader):
        if torch.cuda.is_available():
            spectrograms = spectrograms.cuda()

        out = model.forward(spectrograms)

        loss = criterion(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses[idx] = loss

        print(f'Epoch: [{epoch}][{idx}/{len(data_loader)}]\t'
              f'Loss: {loss} ({torch.mean(losses[:idx+1])})')


if __name__ == "__main__":
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_librispeech = utils.load_train()
    valid_librispeech = utils.load_validation()
    test_librispeech = utils.load_test()


    if os.path.isfile('./data/processed/train.pt'):
        train_data = torch.load('./data/processed/train.pt')
    else:
        train_data = utils.preprocess_data(train_librispeech,
                                           REMOVED_SPEAKERS,
                                           params['num_speakers'],
                                           params['num_utterances'],
                                           params['waveform_length'],
                                           'train.pt')
    #TODO:
    # preprocess validation and test data as well, need to find which speakers need to be removed

    spec_size = (params['spec_length'], params['spec_channels'])
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                device,
                                                                                                spec_size))

    #TODO:
    # initialize validation_loader and test_loader

    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])
    optimizer = torch.optim.SGD(encoder.parameters(), params['learning_rate'])
    criterion = EndToEndLoss(10.0, -5.0, device)

    best_loss = sys.maxsize
    best_model = None
    for epoch in range(params['epochs']):
        train(epoch, train_loader, encoder, optimizer, criterion)
        #TODO:
        # 1. add in validation function and make a call to it
        # 2. if valid_loss < best_loss, update best_loss and deepcopy the model
        # 3. print out best loss
        # 4. save model checkpoint

    #TODO:
    # plot visualizations for test data?

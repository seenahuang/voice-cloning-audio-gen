import utils
from models.speaker_encoder import SpeakerEncoder
from models.EndToEndLoss import EndToEndLoss
import torch
from torch.utils.data import DataLoader
import os
import sys
import copy
import json

REMOVED_SPEAKERS_TRAINING = {1992, 8014, 7312, 445, 1183} # speaker IDs that have fewer than 80 utterances for training


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

        print(f'Epoch: [{epoch}][{idx+1}/{len(data_loader)}]\t'
              f'Training Loss: {loss} ({torch.mean(losses[:idx+1])})')

    return torch.mean(losses)

def validate(epoch, data_loader, model, criterion):
    losses = torch.zeros(len(data_loader))

    for idx, spectrograms in enumerate(data_loader):

        if torch.cuda.is_available():
            spectrograms = spectrograms.cuda()

        with torch.no_grad():
            out = model.forward(spectrograms)
            loss = criterion(out)

        losses[idx] = loss

    print(f'Epoch: [{epoch}]\t'
          f'Validation Loss: {torch.mean(losses)}\n')

    return torch.mean(losses)


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
                                           REMOVED_SPEAKERS_TRAINING,
                                           params['train_num_speakers'],
                                           params['train_num_utterances'],
                                           params['train_waveform_length'],
                                           'train')

    if os.path.isfile('./data/processed/validate.pt'):
        validation_data = torch.load('.data/processed/validation.pt')
    else:
        validation_data = utils.preprocess_data(valid_librispeech,
                                                {},
                                                params['validate_num_speakers'],
                                                params['validate_num_utterances'],
                                                params['validate_waveform_length'],
                                                'validation')

    train_spec_size = (params['train_spec_length'], params['spec_channels'])
    validate_spec_size = (params['validate_spec_length'], params['spec_channels'])
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                device,
                                                                                                train_spec_size))

    validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                device,
                                                                                                validate_spec_size))

    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])
    optimizer = torch.optim.SGD(encoder.parameters(), params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['decay_step'], gamma=params['decay_gamma'])
    criterion = EndToEndLoss(10.0, -5.0, device)

    with open('speaker_encoder_loss.json', 'r') as f:
        loss_dict = json.load(f)
        overall_best_loss = loss_dict['best_loss']

    best_model = None
    best_loss = sys.maxsize

    train_losses = []
    val_losses = []

    for epoch in range(params['epochs']):
        curr_train_loss = train(epoch, train_loader, encoder, optimizer, criterion)
        train_losses.append(curr_train_loss.item())

        curr_val_loss = validate(epoch, validation_loader, encoder, criterion)
        val_losses.append(curr_val_loss.item())

        if curr_val_loss < best_loss:
            best_loss = curr_val_loss
            best_model = copy.deepcopy(encoder)

        scheduler.step()

    print(f'Best Loss: {best_loss}\n')

    if best_loss < overall_best_loss:
        torch.save(best_model.state_dict(), 'checkpoints/speaker_encoder/speaker_encoder.pth')

        loss_dict['best_loss'] = int(best_loss)
        with open('speaker_encoder_loss.json', "r+") as f:
            f.seek(0)
            f.write(json.dumps(loss_dict))
            f.truncate()

        utils.plot_curves(range(params['epochs']), train_losses, val_losses)
        print('Found better model')
    else:
        print('Did not find better model')

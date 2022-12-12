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

    best_loss = sys.maxsize
    best_model = None
    best_train_losses = []
    best_val_losses = []
    best_params = {}

    batch_sizes = [4, 8, 16]
    learning_rates = [i/100 for i in range(1, 12, 2)]
    decay_gammas = [i/100 for i in range(10, 71, 15)]

    i = 1
    total_steps = len(batch_sizes) * len(learning_rates) * len(decay_gammas)
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for decay_gamma in decay_gammas:
                better_model_found = False

                print(f"\n----------Step {i}/{total_steps}----------\n")
                print(f"Batch Size: {batch_size}\nLearning Rate: {learning_rate}\n"
                      f"Decay Gamma: {decay_gamma}\n")

                train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                            device,
                                                                                                            train_spec_size))

                validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                                 device,
                                                                                                                 validate_spec_size))

                encoder = SpeakerEncoder(params['input_size'],
                                         params['hidden_size'],
                                         device,
                                         num_layers=params['num_layers'],
                                         embedding_size=params['embedding_size'])

                optimizer = torch.optim.SGD(encoder.parameters(), learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params['decay_step'], decay_gamma)
                criterion = EndToEndLoss(10.0, -5.0, device)

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
                        better_model_found = True

                    scheduler.step()

                if better_model_found:
                    best_train_losses = train_losses
                    best_val_losses = val_losses
                    best_params = {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "decay_gamma": decay_gamma,
                        "loss": int(best_loss)
                    }

                i += 1

    torch.save(best_model.state_dict(), 'checkpoints/speaker_encoder/speaker_encoder.pth')
    utils.plot_curves(range(params['epochs']), best_train_losses, best_val_losses)

    with open('best_speaker_encoder.json', "r+") as f:
        f.seek(0)
        f.write(json.dumps(best_params), indent=4)
        f.truncate()

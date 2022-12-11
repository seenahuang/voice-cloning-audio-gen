import utils
from models.speaker_encoder import SpeakerEncoder
from models.EndToEndLoss import EndToEndLoss
import torch
from torch.utils.data import DataLoader
import os
import sys
import copy
import json
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

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


def train_encoder(config, params=None, device=torch.device('cpu'), checkpoint_dir=None):
    # initialize/load model
    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])

    optimizer = torch.optim.SGD(encoder.parameters(), config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['decay_step'], gamma=config['decay_gamma'])
    criterion = EndToEndLoss(10.0, -5.0, device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        encoder.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # load data
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
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                device,
                                                                                                train_spec_size))

    validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.generate_spectrograms(x,
                                                                                                device,
                                                                                                validate_spec_size))

    for epoch in range(params['epochs']):
        running_loss = 0.0
        epoch_steps = 0
        for i, spectrograms in enumerate(train_loader):
            if torch.cuda.is_available():
                spectrograms = spectrograms.cuda()

            optimizer.zero_grad()

            out = encoder.forward(spectrograms)
            loss = criterion(out)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/epoch_steps}")

        val_loss = 0.0
        val_steps = 0
        for i, spectrograms in enumerate(validation_loader):
            if torch.cuda.is_available():
                spectrograms = spectrograms.cuda()

            with torch.no_grad():
                out = encoder.forward(spectrograms)
                loss = criterion(out)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((encoder.state_dict(), optimizer.state_dict()), path)

        scheduler.step()
        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")


def main(num_samples=10, max_num_epochs=50):
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        "batch_size": tune.choice([4,8,16]),
        "learning_rate": tune.loguniform(1e-2, 2e-1),
        "decay_gamma": tune.loguniform(0.1, 1),
        "decay_step": tune.choice([2, 5, 10])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(metric_columns=["loss"])

    result = tune.run(
        partial(train_encoder, params=params, device=device),
        resources_per_trial={"cpu": 12, "gpu":1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial loss: {best_trial.last_result['loss']}")

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    torch.save(model_state, 'checkpoints/best_speaker_encoder.pth')

if __name__ == "__main__":
    main()

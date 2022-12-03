import torchaudio
import torch
import os
from dataloader import SpecGen
from yaml import safe_load
import math


def load_train():
    if not os.path.isdir('./data'):
        os.makedirs('./data')

    return torchaudio.datasets.LIBRISPEECH("./data",
                                           url="train-clean-100",
                                           download=not os.path.isdir('./data/LibriSpeech/train-clean-100'))


def load_validation():
    if not os.path.isdir('./data'):
        os.makedirs('./data')

    return torchaudio.datasets.LIBRISPEECH("./data",
                                           url="dev-clean",
                                           download=not os.path.isdir('./data/LibriSpeech/dev-clean'))


def load_test():
    if not os.path.isdir('./data'):
        os.makedirs('./data')

    return torchaudio.datasets.LIBRISPEECH("./data",
                                           url="test-clean",
                                           download=not os.path.isdir('./data/LibriSpeech/test-clean'))


def generate_spectrograms(data, device):
    spec_gen = SpecGen()
    spec_gen.to(device)
    #TODO: figure out how to make spectrograms the same size
    spec_size = 268

    spectrograms = torch.empty((len(data), spec_size, 40))
    speakers = torch.empty((len(data)))
    for idx, (waveform, _, _, speaker, _, _) in enumerate(data):
        spectrogram = spec_gen(waveform.to(device))
        if spectrogram.shape[1] < spec_size:
            pad_amount = spec_size - spectrogram.shape[1]
            first_pad = math.ceil(pad_amount/2)
            second_pad = math.floor(pad_amount/2)
            spectrogram = torch.nn.functional.pad(input=spectrogram, pad=(first_pad, second_pad), value=0)
        spectrograms[idx] = torch.transpose(spectrogram, 0, 1)
        speakers[idx] = speaker

    return spectrograms, speakers


def retrieve_hyperparams(config_file_name):
    with open(f'./configs/{config_file_name}') as f:
        config = safe_load(f)

    params = {}
    for key in config:
        for k, v in config[key].items():
            params[k] = v

    return params


def train(epoch, data_loader, model, optimizer, criterion):
    for idx, (spectrograms, speakers) in enumerate(data_loader):
        if torch.cuda.is_available():
            spectrograms = spectrograms.cuda()
            speakers = speakers.cuda()

        out = model.forward(spectrograms)
        # TODO: make sure loss forward method uses embeddings, speaker separated?
        loss = criterion(out, speakers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #TODO: calculate performance metric and print out update?
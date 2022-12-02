import torchaudio
import numpy as np
import os
from dataloader import SpecGen
from yaml import safe_load


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

    spectrograms = []
    speakers = []
    for waveform, _, _, speaker, _, _ in data:
        spectrogram = spec_gen(waveform.to(device))
        spectrograms.append(spectrogram)
        speakers.append(speaker)

    return speakers, spectrograms


def retrieve_hyperparams(config_file_name):
    with open(f'./configs/{config_file_name}') as f:
        config = safe_load(f)

    params = {}
    for key in config:
        for k, v in config[key].items():
            params[k] = v

    return params


def train(epoch, data_loader, model, criterion, optimizer):
    for batch, spec_speakers in enumerate(data_loader):
        spec_speakers = sorted(spec_speakers, key=lambda x: x[0])
        spectrograms = [x[1] for x in spec_speakers]
        speakers = [x[0] for x in spec_speakers]


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


def generate_spectrograms(data_type, device):
    if data_type == 'train':
        data = load_train()
    elif data_type == 'test':
        data = load_test()
    else:
        raise Exception("Invalid data type, must be 'train' or 'test'")

    spec_gen = SpecGen()
    spec_gen.to(device)

    speaker_to_spec = {}
    for waveform, _, text, speaker, _, _ in data:
        spectrogram = spec_gen(waveform.to(device))
        if speaker in speaker_to_spec:
            speaker_to_spec[speaker].append(spectrogram)
        else:
            speaker_to_spec[speaker] = [spectrogram]

    return speaker_to_spec


def retrieve_hyperparams(config_file_name):
    with open(f'./configs/{config_file_name}') as f:
        config = safe_load(f)

    params = {}
    for key in config:
        for k, v in config[key].items():
            params[k] = v

    return params

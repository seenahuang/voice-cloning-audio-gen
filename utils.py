import torchaudio
import torch
import os
import dataloader


def load_train():
    if not os.path.isdir('./data'):
        os.makedirs('./data')

    return torchaudio.datasets.LIBRISPEECH("./data",
                                           url="train-clean-100",
                                           download=not os.path.isdir('./data/LibriSpeech/train-clean-100'))


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

    spec_gen = dataloader.SpecGen()
    spec_gen.to(device)

    spectrograms = []
    for waveform, _, text, speaker, _, _ in data:
        spectrogram = spec_gen(waveform.to(device)).squeeze(0)
        spectrograms.append(spectrogram)

    return spectrograms
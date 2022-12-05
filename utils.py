import torchaudio
import os
from dataloader import SpecGen
from yaml import safe_load
import torch
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


def generate_spectrograms(data, device, spec_size):
    spec_gen = SpecGen()
    spec_gen.to(device)

    spectrograms = torch.empty((len(data), data[0].shape[0], spec_size[0], spec_size[1]))

    for idx, waveforms in enumerate(data):
        spectrogram = spec_gen(waveforms.to(device))
        spectrograms[idx] = torch.transpose(spectrogram, 1, 2)

    return spectrograms.flatten(2, 3)


def retrieve_hyperparams(config_file_name):
    with open(f'./configs/{config_file_name}') as f:
        config = safe_load(f)

    params = {}
    for key in config:
        for k, v in config[key].items():
            params[k] = v

    return params


def preprocess_data(librispeech, removed_speakers, num_speakers, num_utterances, waveform_length, data_type):
    final_data = torch.empty((num_speakers, num_utterances, waveform_length))
    speaker_to_index = {}
    furthest_index = 0
    for waveform, _, _, speaker, _, _ in librispeech:
        if speaker in removed_speakers:
            continue

        if speaker not in speaker_to_index:
            speaker_to_index[speaker] = (furthest_index, 0)
            curr_index = furthest_index
            furthest_index += 1
            m = 0
        else:
            curr_index, m = speaker_to_index[speaker]
            if m >= 80:
                continue
            speaker_to_index[speaker] = (curr_index, m+1)

        if waveform.shape[1] < waveform_length:
            total_pad = waveform_length-waveform.shape[1]
            l_pad = math.ceil(total_pad / 2)
            r_pad = math.floor(total_pad / 2)
            final_waveform = torch.nn.functional.pad(waveform, (l_pad, r_pad))
        else:
            total_slice = waveform.shape[1] - waveform_length
            l_slice = math.ceil(total_slice / 2)
            r_slice = math.floor(total_slice / 2)
            final_waveform = waveform[0, l_slice:waveform.shape[1]-r_slice].unsqueeze(0)

        final_data[curr_index, m] = final_waveform

    torch.save(final_data, f'./data/processed/{data_type}.pt')
    return final_data


def train(epoch, data_loader, model, optimizer, criterion):
    for idx, spectrograms in enumerate(data_loader):
        if torch.cuda.is_available():
            spectrograms = spectrograms.cuda()

        out = model.forward(spectrograms)

        loss = criterion(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #TODO: calculate performance metric and print out update?

import torch
import torchaudio.transforms as T


class SpecGen(torch.nn.Module):
    # implementation of spectrogram generation taken from pytorch documentation
    # https://pytorch.org/audio/stable/transforms.html
    def __init__(self, input_freq=16000, resample_freq=8000, n_fft=1024, n_mel=40, stretch_factor=0.8):
        super(SpecGen, self).__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = T.Spectrogram(n_fft=n_fft, power=2)
        self.spec_aug = torch.nn.Sequential(
            T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=80),
            T.TimeMasking(time_mask_param=80)
        )

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform):
        resampled = self.resample(waveform)
        spec = self.spec(resampled)
        spec = self.spec_aug(spec)
        mel = self.mel_scale(spec).squeeze(0)
        return mel

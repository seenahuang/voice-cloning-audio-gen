import utils
import torchaudio
from models.speaker_encoder import SpeakerEncoder
import torch
from models.denoiser import Denoiser
import IPython.display as ipd
import math

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    sampling_rate = 22050

    test_audio_gen = utils.load_test()
    waveforms = []
    texts = []

    # save 2 utterances from same speaker as wav files
    waveform, sample_rate, text, speaker, _, _ = test_audio_gen[0]
    torchaudio.save(f"audio/speaker_{speaker}_original1.wav", waveform, sample_rate)
    waveforms.append(waveform)
    texts.append(text)

    waveform, sample_rate, text, speaker, _, _ = test_audio_gen[1]
    torchaudio.save(f"audio/speaker_{speaker}_original2.wav", waveform, sample_rate)
    waveforms.append(waveform)
    texts.append(text)

    #TODO:
    # 4. Call generate() method of tacotron with processed text + speaker embedding
    # 5. Take resulting spectrogram and pass through waveglow
    # 6. Save "speaker_0_after.wav" from waveglow

    #initialize speaker encoder
    speaker_encoder_checkpoint_path = "checkpoints/speaker_encoder.pth"
    speaker_encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])
    speaker_encoder.load_state_dict(torch.load(speaker_encoder_checkpoint_path))
    speaker_encoder.eval()

    # generate speaker embedding
    waveform_length = params['validate_waveform_length']
    final_waveforms = torch.empty((1, 2, waveform_length))
    for i, waveform in enumerate(waveforms):
        if waveform.shape[1] < waveform_length:
            total_pad = waveform_length - waveform.shape[1]
            l_pad = math.ceil(total_pad / 2)
            r_pad = math.floor(total_pad / 2)
            final_waveform = torch.nn.functional.pad(waveform, (l_pad, r_pad))
        else:
            total_slice = waveform.shape[1] - waveform_length
            l_slice = math.ceil(total_slice / 2)
            r_slice = math.floor(total_slice / 2)
            final_waveform = waveform[0, l_slice:waveform.shape[1] - r_slice].unsqueeze(0)

        final_waveforms[0, i] = final_waveform

    spectrograms = utils.generate_spectrograms(final_waveforms, device,
                                               (params['validate_spec_length'], params['spec_channels'])).to(device)

    embeddings = speaker_encoder(spectrograms)

    #initialize tacotron
    tacotron_path = "checkpoints/tacotron2.pt"
    tacotron = None


    #initialize waveglow
    waveglow_path = 'checkpoints/waveglow.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    mel_outputs, linear, attn_scores = tacotron.generate(text, embeddings)

    with torch.no_grad():
        audio = waveglow.infer(linear, sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=sampling_rate)

    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    ipd.Audio(audio_denoised.cpu().numpy(), rate=sampling_rate)
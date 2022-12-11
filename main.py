import utils
import torchaudio
from models.speaker_encoder import SpeakerEncoder
import torch
from models.denoiser import Denoiser
import IPython.display as ipd
import math
from models.tacotron import Tacotron

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

    test_audio_gen = utils.load_test()
    waveforms = []
    texts = []
    sample_rates = []

    # save 2 utterances from same speaker as wav files
    waveform, sample_rate, text, speaker, _, _ = test_audio_gen[0]
    torchaudio.save(f"audio/speaker_{speaker}_original1.wav", waveform, sample_rate)
    waveforms.append(waveform)
    texts.append(processor(text)[0].to(device))
    sample_rates.append(sample_rate)

    waveform, sample_rate, text, speaker, _, _ = test_audio_gen[1]
    torchaudio.save(f"audio/speaker_{speaker}_original2.wav", waveform, sample_rate)
    waveforms.append(waveform)
    texts.append(processor(text)[0].to(device))
    sample_rates.append(sample_rate)

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
    tacotron = Tacotron(embed_dims=512,
                        num_chars=66,
                        encoder_dims=256,
                        decoder_dims=128,
                        n_mels=80,
                        fft_bins=80,
                        postnet_dims=512,
                        encoder_K=5,
                        lstm_dims=1024,
                        postnet_K=5,
                        num_highways=4,
                        dropout=.5,
                        stop_threshold=-3.4,
                        speaker_embedding_size=256).to(device)
    tacotron.load_state_dict(torch.load(tacotron_path)['model_state'])
    tacotron.eval()


    #initialize waveglow
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    # generate wav file given each text/embedding combination
    for i, text in enumerate(texts):
        mel_outputs, linear, attn_scores = tacotron.generate(text)#, embeddings[0,i])

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs, sigma=0.666)

        audio_denoised = denoiser(audio.cpu(), strength=0.01)[:, 0]
        torchaudio.save(f"audio/speaker_{speaker}_gen{i+1}.wav", audio.cpu(), sample_rates[i])
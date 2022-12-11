import utils
import torchaudio
from models.speaker_encoder import SpeakerEncoder
import torch
from outside_code.denoiser import Denoiser
import math
from outside_code.tacotron import Tacotron
from outside_code.inference import Synthesizer

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
    texts.append(text)
    sample_rates.append(sample_rate)

    waveform, sample_rate, text, speaker, _, _ = test_audio_gen[1]
    torchaudio.save(f"audio/speaker_{speaker}_original2.wav", waveform, sample_rate)
    waveforms.append(waveform)
    texts.append(text)
    sample_rates.append(sample_rate)

    #initialize speaker encoder
    speaker_encoder_checkpoint_path = "checkpoints/speaker_encoder/speaker_encoder.pth"
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
    synthesizer = Synthesizer(tacotron_path, verbose=False)

    #initialize waveglow
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    # generate wav file given each text/embedding combination
    for i, text in enumerate(texts):
        mel_outputs = synthesizer.synthesize_spectrograms([text], [embeddings[0,i].detach().cpu()])
        spec = torch.from_numpy(mel_outputs[0]).cuda()
        with torch.no_grad():
            audio = waveglow.infer(torch.unsqueeze(spec, dim=0), sigma=0.666)

        # audio_denoised = denoiser(audio.cpu(), strength=0.01)[:, 0]
        torchaudio.save(f"audio/speaker_{speaker}_gen{i+1}.wav", audio.cpu(), sample_rates[i])
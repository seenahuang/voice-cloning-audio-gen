import utils
import torchaudio
from models.speaker_encoder import SpeakerEncoder
import torch
from models.denoiser import Denoiser
import IPython.display as ipd

if __name__ == "__main__":
    test_audio_gen = utils.load_test()
    waveform, sample_rate, text, speaker, _, _ = test_audio_gen[0]
    torchaudio.save(f"audio/speaker_{speaker}_original.wav", waveform, sample_rate)

    #TODO:
    # 1. Initialize and load speaker encoder and tacotron
    # 2. Generate spectrogram from waveform (preprocess length)
    # 3. Pass spectrogram into speaker encoder to generate speaker embedding
    # 4. Call generate() method of tacotron with processed text + speaker embedding
    # 5. Take resulting spectrogram and pass through waveglow
    # 6. Save "speaker_0_after.wav" from waveglow
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    sampling_rate = 22050

    text = "sample text to generate"

    #initialize speaker encoder
    speaker_encoder_checkpoint_path = "checkpoints/speaker_encoder.pth"
    speaker_encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])
    speaker_encoder.load_state_dict(torch.load(speaker_encoder_checkpoint_path))
    speaker_encoder.eval()

    embeddings = None

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
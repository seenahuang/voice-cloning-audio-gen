import utils
import torchaudio

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

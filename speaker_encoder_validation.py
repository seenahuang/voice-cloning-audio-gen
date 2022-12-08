from models.EndToEndLoss import EndToEndLoss
from models.speaker_encoder import SpeakerEncoder
import utils
import torch


params = utils.retrieve_hyperparams("speaker_encoder.yaml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpeakerEncoder(params['input_size'],
                       params['hidden_size'],
                       device,
                       num_layers=params['num_layers'],
                       embedding_size=params['embedding_size'])
model.load_state_dict(torch.load('checkpoints/speaker_encoder.pth'))
model.eval()

testing_data = utils.load_test()
testing_data = utils.preprocess_data(testing_data,
                                     {},
                                     params['validate_num_speakers'],
                                     params['validate_num_utterances'],
                                     params['validate_waveform_length'],
                                     'test')

data = utils.generate_spectrograms(testing_data, device, (params['validate_spec_length'], params['spec_channels']))
embeddings = model(data.to(device))

criterion = EndToEndLoss(10.0, -5.0, device)
centroids = torch.mean(embeddings, dim=1)
a = criterion.cos_similarity(embeddings, centroids)

print()
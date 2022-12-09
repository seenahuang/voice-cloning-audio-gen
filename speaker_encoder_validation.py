from models.EndToEndLoss import EndToEndLoss
from models.speaker_encoder import SpeakerEncoder
import utils
import torch
import numpy as np
from matplotlib import pyplot as plt


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
                                     params['test_num_speakers'],
                                     params['test_num_utterances'],
                                     params['validate_waveform_length'],
                                     'test')

data = utils.generate_spectrograms(testing_data, device, (params['validate_spec_length'], params['spec_channels']))
embeddings = model(data.to(device))

criterion = EndToEndLoss(10.0, -5.0, device)
centroids = torch.mean(embeddings, dim=1)
similarity_matrix = criterion.cos_similarity(embeddings, centroids).detach().cpu()

y_labels = [f'utterance {i}' for i in range(params['test_num_utterances'])]
x_labels = [f'speaker {i}' for i in range(params['test_num_speakers'])]
for i, speaker_matrix in enumerate(similarity_matrix):
    fig, ax = plt.subplots(figsize=(10,10))
    cax = ax.matshow(speaker_matrix, interpolation='nearest')
    ax.grid(True)
    plt.title(f'Embedding Similarity Matrix Speaker {i}')
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    fig.colorbar(cax, ticks=[i/10 for i in range(10)])
    plt.savefig(f'plots/sim_matrix_{i}.png')
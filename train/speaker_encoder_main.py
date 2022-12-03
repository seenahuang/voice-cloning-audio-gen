import utils
from models.speaker_encoder import SpeakerEncoder
from models.EndToEndLoss import EndToEndLoss
import torch


if __name__ == "__main__":
    params = utils.retrieve_hyperparams("speaker_encoder.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_librispeech = utils.load_train()
    valid_librispeech = utils.load_validation()
    test_librispeech = utils.load_test()

    train_loader = torch.utils.data.DataLoader(dataset=train_librispeech,
                                         batch_size=params['batch_size'],
                                         shuffle=True,
                                         collate_fn=lambda x: utils.generate_spectrograms(x, device))

    encoder = SpeakerEncoder(params['input_size'],
                             params['hidden_size'],
                             device,
                             num_layers=params['num_layers'],
                             embedding_size=params['embedding_size'])
    optimizer = torch.optim.SGD(encoder.parameters(), params['learning_rate'])
    #TODO: initialize E2E loss
    criterion = EndToEndLoss()
    for epoch in range(params['epochs']):
        utils.train(epoch, train_loader, encoder, optimizer, criterion)

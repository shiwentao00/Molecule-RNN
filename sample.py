import torch
import yaml
from dataloader import SELFIEVocab
import selfies as sf
from model import RNN


if __name__ == "__main__":
    # detect cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # load the configuartion file in output
    config_dir = "../results/run_1/config.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # load vocab
    vocab = SELFIEVocab(vocab_path=config['vocab_path'])

    # load model
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    model.load_state_dict(torch.load(
        config['out_dir'] + 'trained_model.pt',
        map_location=torch.device('cpu')))

    # feed the model <sos> and start sampling
    # output sampled SELFIES
    selfies = model.sample(vocab)
    print('Sampled SELFIES: \n', selfies)

    # output sampled SMILES
    smiles = sf.decoder(selfies)
    print('Sampled SMILES: \n', smiles)

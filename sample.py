import argparse
import torch
import yaml
from dataloader import SELFIEVocab, RegExVocab
import selfies as sf
from model import RNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir",
                        help="configuration file to load trained model.")
    args = parser.parse_args()
    config_dir = args.config_dir
    print("configuration file path: ", config_dir)

    # load the configuartion file in output
    #config_dir = "../results/run_1/config.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # detect cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # load vocab
    which_vocab, vocab_path = config["which_vocab"], config["vocab_path"]

    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    # load model
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    model.load_state_dict(torch.load(
        config['out_dir'] + 'trained_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    mol = model.sample(vocab)

    if vocab.name == "selfies":
        mol = sf.decoder(mol)

    print('Sampled SMILES: \n', mol)

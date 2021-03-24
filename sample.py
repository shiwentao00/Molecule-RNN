# Copyright: Wentao Shi, 2021
import argparse
import torch
import yaml
import selfies as sf
import multiprocessing as mp
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN


def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=True,
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    parser.add_argument("-num_samples",
                        required=False,
                        default=1,
                        help="number of samples to generate per process"
                        )
    parser.add_argument("-num_procs",
                        required=False,
                        default=1,
                        help="number of processes to use"
                        )
    return parser.parse_args()


def sample(num_samples):
    """
    Returns a list of sampled SMILES.
    """
    # using process id as random seed of pytorch, such
    # that different processes sample different molecules
    torch.manual_seed(mp.current_process()._identity[0])

    res = []
    for _ in range(num_samples):
        mol = model.sample(vocab)

        if vocab.name == "selfies":
            mol = sf.decoder(mol)

        res.append(mol + "\n")

    return res


if __name__ == "__main__":
    args = get_args()
    result_dir = args.result_dir

    # load the configuartion file in output
    # config_dir = "../results/run_1/config.yaml"
    config_dir = result_dir + "config.yaml"
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
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    # load model
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    model.load_state_dict(torch.load(
        config['out_dir'] + 'trained_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    # initiate multiple processes to sample
    num_samples = int(args.num_samples)
    num_procs = int(args.num_procs)
    print("creating {} processes, each process sampling {} molecules.".format(
        num_procs, num_samples))
    with mp.Pool(num_procs) as p:
        smiles = p.map(sample, [num_samples] * num_procs)

    # wrtite results to file
    out_file = open(result_dir + "sampled_molecules.out", "w")
    for smiles_list in smiles:
        for mol in smiles_list:
            out_file.write(mol)
    out_file.close

# Copyright: Wentao Shi, 2021
import argparse
import torch
import yaml
from tqdm import tqdm
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN


def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=True,
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    parser.add_argument("-batch_size",
                        required=False,
                        default=1024,
                        help="number of samples to generate per mini-batch"
                        )
    parser.add_argument("-num_batches",
                        required=False,
                        default=10,
                        help="number of batches to generate"
                        )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    result_dir = args.result_dir
    batch_size = args.batch_size
    num_batches = args.num_batches

    # load the configuartion file in output
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
        map_location=torch.device(device)))
    model.eval()

    out_file = open(result_dir + "sampled_molecules.out", "w")

    for _ in tqdm(range(num_batches)):
        # sample molecules as integers
        sampled_ints = model.sample(
            batch_size=batch_size,
            vocab=vocab, 
            device=device
        )

        # convert integers back to SMILES
        molecules = []
        sampled_ints = sampled_ints.tolist()
        for ints in sampled_ints:
            molecule = []
            for x in ints:
                if vocab.int2tocken[x] == '<eos>':
                    break
                else:
                    molecule.append(vocab.int2tocken[x])
            molecules.append("".join(molecule))

        # save the sampled SMILES to output file
        for mol in molecules:
            out_file.write(mol + '\n')

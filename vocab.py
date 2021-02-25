"""
Generate the vocabulary of the selfies of the smiles in the dataset
"""
import yaml
from os import listdir
from os.path import isfile, join
import selfies as sf


def read_smiles_file(path, percentage):
    with open(path, 'r') as f:
        smiles = [line.split(' ')[0] for line in f.readlines()]
    num_data = len(smiles)
    return smiles[0:int(num_data * percentage)]


if __name__ == "__main__":
    dataset_dir = "../zinc-smiles/"
    output_vocab = "../vocab.yaml"

    smiles_files = [f for f in listdir(
        dataset_dir) if isfile(join(dataset_dir, f))]

    all_selfies = []
    for f in smiles_files:
        smiles = read_smiles_file(dataset_dir + f, 1)
        selfies = [sf.encoder(x) for x in smiles if sf.encoder(x) is not None]
        all_selfies.extend(selfies)

    vocab = sf.get_alphabet_from_selfies(all_selfies)
    vocab.add('<sos>')
    vocab.add('<eos>')

    vocab_dict = {}
    for i, token in enumerate(vocab):
        # reserve 0 for padding
        vocab_dict[token] = i + 1

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)

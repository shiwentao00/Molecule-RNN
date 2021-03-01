"""generate the vocabulary accorrding to the regular expressions of
SMILES of molecules."""
import yaml
from os import listdir
from os.path import isfile, join
import re


def read_smiles_file(path, percentage):
    with open(path, 'r') as f:
        smiles = [line.split(' ')[0] for line in f.readlines()[1:]]
    num_data = len(smiles)
    return smiles[0:int(num_data * percentage)]


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    return tokenized


if __name__ == "__main__":
    dataset_dir = "../../zinc-smiles/"
    output_vocab = "./regex_vocab.yaml"

    smiles_files = [f for f in listdir(
        dataset_dir) if isfile(join(dataset_dir, f))]
    all_tokens = []
    for i, f in enumerate(smiles_files):
        smiles = read_smiles_file(dataset_dir + f, 1)
        tokens = []
        [tokens.extend(tokenize(x)) for x in smiles]
        all_tokens.extend(tokens)
        print('{} out of {} files processed.'.format(i, len(smiles_files)))

    all_tokens = set(all_tokens)

    vocab_dict = {}
    for i, token in enumerate(all_tokens):
        vocab_dict[token] = i

    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)

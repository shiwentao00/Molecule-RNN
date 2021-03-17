"""generate the vocabulary accorrding to the regular expressions of
SMILES of molecules."""
import yaml
import re
from tqdm import tqdm


def read_smiles_file(path, percentage):
    with open(path, 'r') as f:
        smiles = [line.strip("\n") for line in f.readlines()]
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
    '[*]' as one token."""
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
    dataset_dir = "../../chembl-data/chembl_28/chembl_28_sqlite/chembl28-cleaned.smi"
    output_vocab = "./chembl_regex_vocab.yaml"

    # read smiles as strings
    smiles = read_smiles_file(dataset_dir, 1)

    print("computing token set...")
    tokens = []
    [tokens.extend(tokenize(x)) for x in tqdm(smiles)]
    tokens = set(tokens)
    print("finish.")

    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i

    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)

import torch
import re
import yaml
import selfies as sf

from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
from torch.nn.utils.rnn import pad_sequence
from pad_idx import PADDING_IDX


def dataloader_gen(dataset_dir, percentage, which_vocab, vocab_path, batch_size, shuffle, drop_last=False):
    """
    Genrate the dataloader for training
    """
    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    dataset = SMILESDataset(dataset_dir, percentage, vocab)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    return dataloader, len(dataset)


def pad_collate(batch):
    """
    Put the sequences of different lengths in a minibatch by paddding.
    """

    lengths = [len(x) for x in batch]

    batch = [torch.tensor(x) for x in batch]

    # use any ingeter that is not in vocab as padding
    x_padded = pad_sequence(batch, batch_first=True, padding_value=PADDING_IDX)

    return x_padded, lengths


class SMILESDataset(Dataset):
    def __init__(self, smiles_file, percentage, vocab):
        """
        smiles_file: path to the .smi file containing SMILES.
        percantage: percentage of the dataset to use.
        """
        super(SMILESDataset, self).__init__()
        assert(0 < percentage <= 1)

        self.percentage = percentage
        self.vocab = vocab

        # load eaqual portion of data from each tranche
        self.data = self.read_smiles_file(smiles_file)
        print("total number of SMILES loaded: ", len(self.data))

        # convert the smiles to selfies
        if self.vocab.name == "selfies":
            self.data = [sf.encoder(x)
                         for x in self.data if sf.encoder(x) is not None]
            print("total number of valid SELFIES: ", len(self.data))

    def read_smiles_file(self, path):
        # need to exclude first line which is not SMILES
        with open(path, "r") as f:
            smiles = [line.strip("\n") for line in f.readlines()]
        num_data = len(smiles)
        return smiles[0:int(num_data * self.percentage)]

    def __getitem__(self, index):
        mol = self.data[index]

        # convert the data into integer tokens
        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.data)


class RegExVocab:
    def __init__(self, vocab_path):
        self.name = "regex"

        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        # a dictionary to map integer back to SMILES
        # tokens for sampling
        self.int2tocken = {}
        for token, num in self.vocab.items():
            if token == "R":
                self.int2tocken[num] = "Br"
            elif token == "L":
                self.int2tocken[num] = "Cl"
            else:
                self.int2tocken[num] = token

    def tokenize_smiles(self, smiles):
        """Takes a SMILES string and returns a list of tokens.
        This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
        '[xx]' as one token."""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = ['<sos>']
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('<eos>')
        # convert tokens to integer tokens
        tokenized = [self.vocab[token] for token in tokenized]
        return tokenized

    def replace_halogen(self, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        string = br.sub('R', string)
        string = cl.sub('L', string)

        return string

    def combine_list(self, smiles):
        return "".join(smiles)


class SELFIEVocab:
    def __init__(self, vocab_path):
        self.name = "selfies"

        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {value: key for key, value in self.vocab.items()}

    def tokenize_smiles(self, mol):
        """convert the smiles to selfies, then return 
        integer tokens."""
        ints = [self.vocab['<sos>']]
        #encoded_selfies = sf.encoder(smiles)
        selfies_list = list(sf.split_selfies(mol))
        for token in selfies_list:
            ints.append(self.vocab[token])
        ints.append(self.vocab['<eos>'])

        return ints

    def combine_list(self, selfies):
        return "".join(selfies)

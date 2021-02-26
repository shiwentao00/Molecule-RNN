import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import selfies as sf
import yaml
from torch.nn.utils.rnn import pad_sequence


def dataloader_gen(dataset_dir, percentage, vocab_path, batch_size, shuffle, drop_last=False):
    """
    Genrate the dataloader for training
    """
    vocab = SELFIEVocab(vocab_path)
    dataset = SMILESDataset(dataset_dir, percentage, vocab)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    return dataloader, len(dataset)


def pad_collate(batch):
    """
    Put the sequences of different lengths in a minibatch by paddding.
    """
    global PADDING_IDX

    lengths = [len(x) for x in batch]

    batch = [torch.tensor(x) for x in batch]

    # use any ingeter that is not in vocab as padding
    x_padded = pad_sequence(batch, batch_first=True, padding_value=PADDING_IDX)

    return x_padded, lengths


class SMILESDataset(Dataset):
    def __init__(self, dataset_dir: str, percentage: float, vocab):
        """
        dataset_dir: directory of the dataset downloaded from Zinc
        percantage: percentage of the dataset to use
        """
        super(SMILESDataset, self).__init__()
        assert(0 < percentage <= 1)

        self.percentage = percentage
        self.smiles_files = [f for f in listdir(
            dataset_dir) if isfile(join(dataset_dir, f))]
        self.vocab = vocab

        # load eaqual portion of data from each tranche
        self.data = []
        for f in self.smiles_files:
            self.data.extend(self.read_smiles_file(dataset_dir + f))
        print("total number of SMILES loaded: ", len(self.data))

        # convert the smiles to selfies
        self.data = [sf.encoder(x)
                     for x in self.data if sf.encoder(x) is not None]
        print("total number of valid SELFIES: ", len(self.data))

    def read_smiles_file(self, path: str):
        with open(path, 'r') as f:
            smiles = [line.split(' ')[0] for line in f.readlines()]
        num_data = len(smiles)
        return smiles[0:int(num_data * self.percentage)]

    def __getitem__(self, index: int):
        mol = self.data[index]
        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.data)


class SELFIEVocab:
    def __init__(self, vocab_path) -> None:
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

    def list2selfies(self, selfies):
        return "".join(selfies)

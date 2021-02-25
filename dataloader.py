from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import selfies as sf
import yaml


def dataloader_gen(dataset_dir, percentage, vocab_path, batch_size, shuffle, drop_last=False):
    """
    Genrate the dataloader for training
    """
    vocab = SELFIEVocab(vocab_path)
    dataset = SMILESDataset(dataset_dir, percentage, vocab)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, len(dataset)


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

        # load eaqual portion of data from each tranche
        self.smiles = []
        for f in self.smiles_files:
            self.smiles.extend(self.read_smiles_file(dataset_dir + f))

        print("total number of SMILES loaded: ", len(self.smiles))

        self.vocab = vocab

    def read_smiles_file(self, path: str):
        with open(path, 'r') as f:
            smiles = [line.split(' ')[0] for line in f.readlines()]
        num_data = len(smiles)
        return smiles[0:int(num_data * self.percentage)]

    def __getitem__(self, index: int):
        mol = self.smiles[index]

        # convert SMILES to SELFIES, then conert to integers
        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.smiles)


class SELFIEVocab:
    def __init__(self, vocab_path) -> None:
        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

    def tokenize_smiles(self, smiles):
        """convert the smiles to selfies, then return 
        integer tokens."""
        ints = [self.vocab['<sos>']]
        encoded_selfies = sf.encoder(smiles)
        selfies_list = list(sf.split_selfies(encoded_selfies))
        for token in selfies_list:
            ints.append(self.vocab[token])
        ints.append(self.vocab['<eos>'])

        return ints


if __name__ == "__main__":
    ds = SMILESDataset('../zinc-smiles/', 0.01)

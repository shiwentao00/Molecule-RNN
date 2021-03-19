# Copyright: Wentao Shi, 2021
import argparse
from rdkit import Chem


def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=True,
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    result_dir = args.result_dir
    smiles_dir = result_dir + "sampled_molecules.out"

    # read smiles
    with open(smiles_dir, "r") as f:
        smiles = [line.strip("\n") for line in f.readlines()]

    num_valid, num_invalid = 0, 0
    for mol in smiles:
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            num_invalid += 1
        else:
            num_valid += 1

    print("sampled {} valid SMILES out of {}, success rate: {}".format(
        num_valid, num_valid + num_invalid, num_valid / (num_valid + num_invalid)))

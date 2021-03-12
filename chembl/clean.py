"""
Pre-process the Chembl dataset:
    1. Normalize the molecule.
    2. Convert the SMILES to canonical form.
"""
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
RDLogger.DisableLog('rdApp.*')


class MolCleaner:
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()

    def process(self, mol):
        mol = Chem.MolFromSmiles(mol)

        if mol is not None:
            mol = self.normarizer.normalize(mol)
            mol = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return mol
        else:
            return None


if __name__ == "__main__":
    in_path = "../../chembl-data/chembl_28/chembl_28_sqlite/chembl28.smi"
    out_path = "../../chembl-data/chembl_28/chembl_28_sqlite/chembl28-cleaned.smi"
    with open(in_path, 'r') as f:
        smiles = [line.strip("\n") for line in f.readlines()]
    f.close()
    print("number of SMILES before cleaning:", len(smiles))

    # clean the molecules
    cleaner = MolCleaner()
    processed = []
    for mol in tqdm(smiles):
        mol = cleaner.process(mol)
        if mol is not None and 20 < len(mol) < 120:
            processed.append(mol)
    print("number of SMILES after cleaning:", len(processed))

    with open(out_path, "w") as f:
        for mol in processed:
            f.write(mol + "\n")
    f.close()

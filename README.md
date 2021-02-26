# Molecule-RNN
Molecule-RNN is a recurrent neural network built with Pytorch to generate molecules for drug discovery. It is trained with the [Zinc](https://zinc.docking.org/) dataset.

## Training
1. Dowdload the SMILES files of molecules from [Zinc](https://zinc.docking.org/). Select "SMILES(*.smi)" and "Flat" opitions when downloading.

2. Modify the path of dataset in ```train.yaml``` to your downloaded dataset by setting the value of ```dataset_dir```.

3. Run the training script.
```
python train.py
```

The training loss:

## Sampling
We can generate molecules by sampling the model according to the output distribution.
```
python sample.py
```
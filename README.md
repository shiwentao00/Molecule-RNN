# Molecule-RNN
Molecule-RNN is a recurrent neural network built with Pytorch to generate molecules for drug discovery. It is trained with the [Zinc](https://zinc.docking.org/) dataset. The [SELFIES](https://github.com/aspuru-guzik-group/selfies) is used as the representation of molecules. The SMILES files are converted to SELFIES during training on-the-fly.

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
The sampled output is in the format of SELFIES:
```
```
Note that the SELFIES is an [automata](https://en.wikipedia.org/wiki/Automata_theory), and it terminates when there is no chemical bonds to build. So the converted SMILES could be shorter than the SELFIES:
```
```

The advantage of SELFIES is that the output is always a valid molecule:

# Molecule-RNN
Molecule-RNN is a recurrent neural network built with Pytorch to generate molecules for drug discovery. 

## Tokenization of SMILES
There are different ways to tokenize SMILES, 3 of them are implemented in this project:
1. Character-level tokenization.
2. Regular expression-based tokenization.
3. SELFIES tokenization.

## Dataset
The chembl28 dataset is used. 

## Training
1. Modify the path of dataset in ```train.yaml``` to your downloaded dataset by setting the value of ```dataset_dir```.

3. Run the training script.
```
python train.py
```

## Sampling
We can generate molecules by sampling the model according to the output distribution. 
```
python sample.py
```


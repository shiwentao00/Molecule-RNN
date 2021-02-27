# the padding index (last index) for batching
import yaml

config_dir = "./train.yaml"
with open(config_dir, 'r') as f:
    config = yaml.full_load(f)
rnn_config = config['rnn_config']
global PADDING_IDX
PADDING_IDX = rnn_config['num_embeddings'] - 1
print("padding index: ", PADDING_IDX)

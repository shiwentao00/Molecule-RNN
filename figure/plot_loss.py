import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=True,
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    return parser.parse_args()


args = get_args()
result_dir = args.result_dir
with open(result_dir + "loss.yaml", "r") as f:
    loss = yaml.full_load(f)
fig, ax = plt.subplots()
ax = sns.lineplot(data=loss, dashes=False)

# find minimum validaiton loss
min_val_loss = np.min(loss)
pos = np.nonzero(loss == min_val_loss)[0]
plt.axvline(x=pos, linewidth=2, color='grey', linestyle='--')
plt.text(pos-35, min_val_loss+0.05, str(min_val_loss)[0:6])

ax.set(xlabel="Epoch")
ax.set(ylabel="Loss")
plt.savefig(result_dir + 'loss.png', bbox_inches="tight")

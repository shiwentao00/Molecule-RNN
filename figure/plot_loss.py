import argparse
import yaml
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
ax.set(xlabel="Epoch")
ax.set(ylabel="Loss")
plt.savefig(result_dir + 'loss.png', bbox_inches="tight")

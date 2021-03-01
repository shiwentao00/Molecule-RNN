import yaml
import matplotlib.pyplot as plt
import seaborn as sns
result_dir = "../../results/run_2/"
with open(result_dir + "loss.yaml", "r") as f:
    loss = yaml.full_load(f)
fig, ax = plt.subplots()
ax = sns.lineplot(data=loss, dashes=False)
ax.set(xlabel="Epoch")
ax.set(ylabel="Loss")
plt.savefig(result_dir + 'loss.png', bbox_inches="tight")

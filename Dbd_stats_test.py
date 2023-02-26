# importing pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dbd_stats.csv")

# drop invalid line number
df = df.drop([26])
a=1

q_vals = df.iloc[:, [4, 9, 10]].values
dh = q_vals[:, np.newaxis, 0]
gens = q_vals[:, np.newaxis, 1]
died_as = q_vals[:, np.newaxis, 2]


# make hist plots from all quantitative data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
ax1.set_ylabel("times it happened")
ax1.hist(dh)
ax1.set_title("number of deadhards")
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.hist(gens)
ax2.set_title("number of gens done")
ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax3.hist(died_as)
ax3.set_title("number I died as")
ax3.set_xticks([1, 2, 3, 4, 5])
plt.show()

bin_vals = df.iloc[:, [3, 5, 6, 7, 8, 11, 12]].values
xs = np.arange(bin_vals.shape[0]) + 1

# make cumulatative sum plot from the binary data
names_list_bin = ["dcs", "suicides", "tunnels", "camps", "escapes", "eruptions", "pain_res"]
fig1, axs = plt.subplots(7, 1, figsize=(8, 12), sharex=True)
for i in range(7):
    axs[i].set_xticks(xs)
    axs[i].plot(xs, np.cumsum(bin_vals[:, i].astype(int)))
    axs[i].set_ylabel(names_list_bin[i])
    axs[i].grid()
    # axs[i].set_xlims((1, xs[-1]))
axs[6].set_xlabel("times it happened total")
plt.show()

# importing pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn import linear_model # needs installed scikit-learn to work
df = pd.read_csv("dbd_stats.csv")

# drop invalid line number
df = df.drop([26])
# set appropriate data type for wrong read-ins
df = df.astype({'dh': 'int32', 'gens': 'int32', 'died as': 'int32', 'escape': 'int32'})

# create new dataframe for values where 'pointrank' was introduced
df_new = df.iloc[12:]
df_new = df_new.astype({'pointrank': 'int32'})

# commands for working with the dataframe
# df.dtypes['dh']
# df.info
# df. describe()
# df.describe()[['dh', 'gens', 'died as', 'escape']]
# df_new.describe()['pointrank']

q_vals = df.iloc[:, [4, 9, 10]].values
dh = q_vals[:, np.newaxis, 0]
gens = q_vals[:, np.newaxis, 1]
died_as = q_vals[:, np.newaxis, 2]


# make hist plots from all quantitative data
# create grid and int values for plots (incorporate latex)
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
    axs[i].plot(xs, np.cumsum(bin_vals[:, i]))
    axs[i].set_ylabel(names_list_bin[i])
    axs[i].grid()
    # axs[i].set_xlims((1, xs[-1]))
axs[6].set_xlabel("times it happened total")
plt.show()

# create x-y type scatterplots of the quantative data
# example with small random offset to show that plotting integer data in scatter is not very helpful
# plt.scatter(gens, dh + 0.1*np.random.rand(30)[:, np.newaxis])

# Reshaped for Logistic function.

X = gens
y = died_as.ravel()
logr = linear_model.LogisticRegression()
logr.fit(X,y)

# predict number that I may die as when the number of gens done is given
predicted = np.zeros(6)
for j in range(6):
    predicted[j] = logr.predict(np.array([float(j)]).reshape(-1,1))
print(f'predicted number I die as with [0, 1, 2, 3, 4, 5] gens done by the time I leave the game \n '
      f' (where 5 means I survive): {predicted}')

# repeat for number of deadhards
X_dh = dh
logr = linear_model.LogisticRegression()
logr.fit(X_dh,y)
predicted_dh = np.zeros(5)
for j in range(5):
    predicted_dh[j] = logr.predict(np.array([float(j)]).reshape(-1,1))
print(f'Predicted number I die as depending on how mandy deadheards [0, 1, 2, 3, 4] are equipped: {predicted_dh}')
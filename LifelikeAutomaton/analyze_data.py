import matplotlib.pyplot as plt
import anndata
import pandas as pd
import numpy as np

files = ["entropy_data.h5ad", "entropy_light_data.h5ad"]
a = [anndata.read(f) for f in files]
X = [np.swapaxes(data.X, 0, 1) for data in a]
xvalues = np.arange(0, 2000+1, 10)

fig, axs = plt.subplots(len(files))
for i in range(len(files)):
    print(a[i].var_names)
    axs[i].set_ylim(0.7, 1.0)
    for j in range(len(X[i])):
        axs[i].plot(xvalues, X[i][j], label=a[i].var_names[j])
    axs[i].legend()

axs[0].set_title("Entropy over Time")
plt.show()

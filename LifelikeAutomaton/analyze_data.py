import matplotlib.pyplot as plt
import anndata
import pandas as pd
import numpy as np

data = np.load("data.npz")
xvalues = data["xvalues"]
states = data["states"]
entropy = data["entropy"]

plt.plot(xvalues, entropy)
plt.title("Average Local Entropy")
plt.show()

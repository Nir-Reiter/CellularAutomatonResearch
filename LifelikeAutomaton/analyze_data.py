import matplotlib.pyplot as plt
import anndata
import pandas as pd
import numpy as np

a = anndata.read("entropy_data.h5ad")
X = np.swapaxes(a.X, 0, 1)
print(X)

xvalues = np.arange(0, 2000+1, 10)
fig, ax = plt.subplots()
lines = [
    ax.plot(xvalues, X[0], label="1.2 1.0 1.0")[0],
    ax.plot(xvalues, X[1], label="1.0 1.0 1.0")[0],
    ax.plot(xvalues, X[2], label="0.8 1.0 1.0")[0],
    ax.plot(xvalues, X[3], label="0.8 1.0 1.2")[0]
]
ax.legend()
plt.show()

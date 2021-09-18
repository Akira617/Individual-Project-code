import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rc('font',family='Times New Roman')

data=pd.read_csv('trainloss.csv',sep=' ',header=None)
data=data.values
data=data[:,3]
plt.figure(figsize=(6,4))
plt.plot(np.arange(len(data)),data*100)
plt.xlabel('iter',fontsize=13)
plt.ylabel('acc',fontsize=13)
plt.tight_layout()
plt.savefig('./acc.png', dpi=300)
plt.show()
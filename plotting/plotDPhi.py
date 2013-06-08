import csv
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("deltaPhiData.csv", delimiter = " ", names=True)

plt.figure()

#plt.hist(data['dPhi'], 100, label=r'$d\phi$', normed=True)
#plt.hist(data['dPhiTip'], 100, label=r'$d\phi_{d_0}$', normed=True)
plt.hist(data['dPhiTip'] + data['dPhi'], 100, label=r'$d\phi_\mathrm{total}$', normed=True)

plt.legend()

plt.show()   
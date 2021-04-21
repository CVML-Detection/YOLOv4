import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,5,100)
y=np.sin(x)
plt.plot(x,y)
plt.title("Plot generated using Matplotlib")
plt.xlabel("x")
plt.ylabel("sinx")
plt.savefig('Customed Plot.png', dpi=300, bbox_inches='tight')
import numpy as np
import matplotlib.pyplot as plt


# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

x = np.array([1, 2, 3, 4, 5])
y1 = np.array([2/15, 1/5, 4/15, 1/15, 1/3])
y2 = np.array([5/35, 7/35, 9/35, 3/35, 11/35])
y3 = np.array([6/40, 8/40, 10/40, 4/40, 12/40])
y4 = np.array([8/50, 10/50, 12/50, 6/50, 14/50])
plt.clf()
plt.plot(x,y1, label='a')
plt.plot(x,y2,label='b')
plt.plot(x,y3,label='c')
plt.plot(x,y4,label='d')
plt.legend()

plt.xlabel("Class Number")
plt.ylabel("Estimated Probability")

plt.show()
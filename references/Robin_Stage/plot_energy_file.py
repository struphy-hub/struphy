from numpy import zeros
import matplotlib.pyplot as plt

energy_file = open('energy.txt', 'r')
content = energy_file.readlines()

N = int(content[0])
dt = float(content[1])

E = zeros((N + 1, 1))
T = list(range(0, N + 1))

for i in range(0, N+1):
	E[i] = float(content[i+2])
	T[i] = T[i]*dt

energy_file.close()

plt.plot(T, E)
axes = plt.gca()
axes.set_ylim(bottom=0)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

d = 0.12 * 1000
L = 2120
El = 8
dof_per_node = 3
n_nodes = El + 1
le = L / El
F = 1000
E = 2 * (10**11) / (10**6)

A = (np.pi * (d/2)**2) 
I = (np.pi * d**4) / 64.0


a1 = E * A / le
a2 = E * I / le**3

k_local = np.array([
    [a1, 0, 0, -a1, 0, 0 ],
    [0, 12*a2, 6*a2, 0, -12*a2, 6*a2],
    [0, 6*a2, 4*a2, 0, -6*a2, 2*a2],
    [-a1, 0, 0, a1, 0, 0],
    [0,-12*a2,-6*a2, 0, 12*a2, -6*a2],
    [0, 6*a2, 2*a2, 0, -6*a2, 4*a2]
])

k_global = np.zeros((n_nodes * dof_per_node, n_nodes * dof_per_node))
F_vector = np.zeros(n_nodes * dof_per_node)

for i in range(El):
    start = 3*i
    end = start + 6
    k_global[start:end, start:end] += k_local

print(k_global)

F_vector[(El // 2) * 3 + 1] = -F
print(F_vector)

k_reduced = np.delete(k_global, [1, -2, -3], axis=0)
k_reduced = np.delete(k_reduced, [1, -2, -3], axis=1)
F_reduced = np.delete(F_vector, [1, -2, -3])

U = np.linalg.solve(k_reduced, F_reduced)
print(U)

U = np.insert(U, 1, [0])
U = np.insert(U, -1, [0, 0])

x = np.linspace(0, L, n_nodes)

plt.plot(x, U[1::3], label='Vertical Displacement')
plt.show()
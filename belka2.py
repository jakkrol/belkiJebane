import numpy as np
import matplotlib.pyplot as plt

b = 0.10 * 1000
h = 0.15 * 1000

L = 2000
n_elem = 10
F = 1000
le = L / n_elem
E = 2 * (10**11) / (10**6)

A = b * h
I = b * h**3 / 12

nodes = n_elem + 1
dof_per_node = 3


a1 = E * A / le
a2 = E * I / le**3

c1 = a1
c2 = 12 * a2
c3 = 6 * a2
c4 = 4 * a2
c5 = 2 * a2

k_local = np.array([
    [ c1,   0,    0,   -c1,    0,    0 ],
    [  0,  c2,   c3,    0,  -c2,   c3 ],
    [  0,  c3,   c4,    0,  -c3,   c5 ],
    [ -c1,   0,    0,    c1,    0,    0 ],
    [  0, -c2,  -c3,    0,   c2,  -c3 ],
    [  0,  c3,   c5,    0,  -c3,   c4 ]
])

print(k_local)

k_global = np.zeros((nodes * dof_per_node, nodes * dof_per_node))
F_vector = np.zeros(nodes * dof_per_node)

for i in range(n_elem):
    start = 3 * i;
    end = start + 6
    k_global[start:end, start:end] += k_local

print(k_global)

F_vector[(n_elem // 2) * 3 + 1] = -F
print(F_vector)


k_reduced = np.delete(k_global, [0,1,-3,-2], axis=0)
k_reduced = np.delete(k_reduced, [0,1,-3,-2], axis=1)
F_reduced = np.delete(F_vector, [0,1,-3,-2])

U = np.linalg.solve(k_reduced, F_reduced)
print(U)

U = np.insert(U, 0, [0,0])
U = np.insert(U, -1, [0,0])

x = np.linspace(0, L, nodes)

plt.plot(x, U[1::3])
plt.plot(x, U[0::3])
plt.plot(x, U[2::3])
plt.show()
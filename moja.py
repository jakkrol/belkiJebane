import numpy as np
import matplotlib.pyplot as plt

L_mm = 2120.0
n_elem = 8

E = 2*(10**11) / (10**6)


d = 0.12 * 1000

l = 2120
le = l / n_elem
F = 1000


I = (np.pi * d**4) / 64.0
A = (np.pi * (d/2)**2) / 4.0


a1 = E * A / le
a2 = E * I / le**3

nodes = n_elem + 1
dof_per_node = 3


k_local = np.array([
    [a1, 0, 0, -a1, 0, 0 ],
    [0, 12*a2, 6*a2, 0, -12*a2, 6*a2],
    [0, 6*a2, 4*a2, 0, -6*a2, 2*a2],
    [-a1, 0, 0, a1, 0, 0],
    [0,-12*a2,-6*a2, 0, 12*a2, -6*a2],
    [0, 6*a2, 2*a2, 0, -6*a2, 4*a2]
])

print(k_local)

k_global = np.zeros((nodes * dof_per_node, nodes * dof_per_node))
F_vector = np.zeros(nodes * dof_per_node)
print(k_global)

for i in range(n_elem):
    start = 3*i
    end = start + 6
    k_global[start:end, start:end] += k_local

print(k_global)


F_vector[len(F_vector) - 2] = -F
print(F_vector)



fixed_dofs = [0, 1, 2]

K_reduced = k_global[3:, 3:]
F_reduced = F_vector[3:]

U = np.linalg.solve(K_reduced, F_reduced)

displacements_w = U[1::3]
print("Displacements: ", displacements_w)


w_analytical_max = (-F * L_mm**3) / (3 * E * I)
print("Analitycznie", w_analytical_max)

print(f"Ugięcie MES na końcu: {displacements_w[-1]} mm")
print(f"Ugięcie analityczne:  {w_analytical_max} mm")


U = np.insert(U, 0, [0,0,0])
size = nodes * dof_per_node
#U2 = np.zeros(size)
#U2[free_dofs] = U
w_mes = U[1::3]
x = np.linspace(0, L_mm, nodes)
print(x)
#plt.figure(figsize=(10, 6))

x_anal_w = np.linspace(0, L_mm, 200)

x_anal_el = (-F * x_anal_w**3) / (3 * E * I)
plt.plot(x_anal_w, x_anal_el)
plt.show()
plt.plot(x, U[0::3])
plt.plot(x, U[1::3])
plt.plot(x, U[2::3])
plt.show()
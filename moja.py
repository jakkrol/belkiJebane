import numpy as np
import matplotlib.pyplot as plt

L_mm = 2000.0       # Długość całkowita [mm] (np. l z tabeli)
n_elem = 10         # Liczba elementów skończonych (np. z tabeli)

E = 2*(10**11) / (10**6)   # Moduł Younga [Pa]

b = 0.10 * 1000            # szerokość [mm]
h = 0.15 * 1000            # wysokość [mm]

l = 2000
le = l / n_elem     # Długość jednego elementu skończonego
F = 1000

I = (b * h**3) / 12.0  
A = b * h

a1 = E * A / le
a2 = E * I / le**3

nodes = n_elem + 1        # Liczba węzłów
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


F_vector[31] = -F   #Nie wiem dlaczego F, a nie -F jak na obrazku ale po porównaniu z analitycznym wychodzi dobrze
print(F_vector)









fixed_dofs = [0, 1, 2]


#all_dofs = np.arange(len(F_vector))
#free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

K_reduced = k_global[3:, 3:]
F_reduced = F_vector[3:]

U = np.linalg.solve(K_reduced, F_reduced)


#U_full = np.zeros(len(F_vector))
#U_full[free_dofs] = U


# displacements_w = U_full[1::3]
displacements_w = U[1::3] 





L = L_mm / 1000.0
w_analytical_max = (-F * L_mm**3) / (3 * E * I)
print("Analitycznie", w_analytical_max)

print(f"Ugięcie MES na końcu: {displacements_w[-1]} mm")
print(f"Ugięcie analityczne:  {w_analytical_max} mm")



size = nodes * dof_per_node
#U2 = np.zeros(size)
#U2[free_dofs] = U
w_mes = U[1::3]
x = np.linspace(0, L_mm, nodes)
print(x)
#plt.figure(figsize=(10, 6))
plt.plot(x[1:], w_mes, 'bo-', label='w mm')
plt.show()
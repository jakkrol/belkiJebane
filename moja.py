import numpy as np
import matplotlib.pyplot as plt

L_mm = 2000.0       # Długość całkowita [mm] (np. l z tabeli)
n_elem = 10         # Liczba elementów skończonych (np. z tabeli)

E = 2*(10**11)            # Moduł Younga [Pa]

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


F_vector[31] = F   #Nie wiem dlaczego F, a nie -F jak na obrazku ale po porównaniu z analitycznym wychodzi dobrze
print(F_vector)









fixed_dofs = [0, 1, 2]

# Tworzymy listę WSZYSTKICH indeksów
all_dofs = np.arange(len(F_vector))

# Wybieramy tylko te, które są WOLNE (nie są utwierdzone)
# Funkcja setdiff1d usuwa z listy 'all_dofs' to, co jest w 'fixed_dofs'
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# --- REDUKCJA MACIERZY ---
# Wycinamy z dużej macierzy tylko ten kawałek, który może się ruszać.
# Używamy np.ix_ do bezpiecznego wycięcia wierszy i kolumn
K_reduced = k_global[np.ix_(free_dofs, free_dofs)]
F_reduced = F_vector[free_dofs]

U = np.linalg.solve(K_reduced, F_reduced)

# U = np.linalg.solve(k_global, F_vector)
# print(U)
# Odtworzenie pełnego wektora przemieszczeń
U_full = np.zeros(len(F_vector))
U_full[free_dofs] = U

# Wyciągnięcie przemieszczeń pionowych (w) dla każdego węzła
# w znajduje się na indeksach: 1, 4, 7, ...
displacements_w = U_full[1::3]






L = L_mm / 1000.0
w_analytical_max = (F * L**3) / (3 * E * I)
w_analytical_max_mm = w_analytical_max * 1000.0
print("Analitycznie", w_analytical_max_mm)

print(f"Ugięcie MES na końcu: {displacements_w[-1]} mm")
print(f"Ugięcie analityczne:  {w_analytical_max_mm} mm")



size = nodes * dof_per_node
U2 = np.zeros(size)
U2[free_dofs] = U
w_mes = U2[1::3] # Przemieszczenia pionowe co 3 indeks
x = np.linspace(0, L_mm, nodes)

plt.figure(figsize=(10, 6))
plt.plot(x, w_mes, 'bo-', label='MES (jednostki mm, bez le)')
#plt.plot(x, w_analityczne, 'r--', label='Teoria')
plt.title(f'Ugięcie belki (obliczenia w mm)\nMax ugięcie: {w_mes[-1]:.3f} mm')
plt.xlabel('x [mm]')
plt.ylabel('w [mm]')
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis()
plt.show()
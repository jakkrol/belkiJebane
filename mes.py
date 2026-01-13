import numpy as np
import matplotlib.pyplot as plt

# --- 1. DANE WEJŚCIOWE (SYMBOLICZNE) ---
# Tutaj wpisz swoje dane z tabeli 
L_mm = 2000.0       # Długość całkowita [mm] (np. l z tabeli)
F_val = 1000.0      # Siła [N] (np. F z tabeli)
n_elem = 10         # Liczba elementów skończonych (np. z tabeli)

# Wybór materiału i przekroju (odkomentuj właściwe)
# Stal: E = 2e11 Pa, Aluminium: E = 7e10 Pa [cite: 313]
E = 2e11            # Moduł Younga [Pa]

# Typ przekroju: 'prostokat' lub 'kolo' [cite: 310, 314]
shape_type = 'prostokat' 

# Wymiary przekroju [m] (uwaga na jednostki w tabeli - tam są w metrach)
b = 0.10            # szerokość [m]
h = 0.15            # wysokość [m]
d = 0.12            # średnica [m] (jeśli koło)

# --- 2. OBLICZENIA WSTĘPNE I GEOMETRYCZNE ---
L = L_mm / 1000.0   # Konwersja długości na metry [m]
le = L / n_elem     # Długość jednego elementu skończonego 

# Obliczenie momentu bezwładności I oraz pola przekroju A
if shape_type == 'prostokat':
    I = (b * h**3) / 12.0  # [cite: 312]
    A = b * h
elif shape_type == 'kolo':
    I = (np.pi * d**4) / 64.0 # [cite: 315]
    A = (np.pi * d**2) / 4.0
else:
    raise ValueError("Nieznany typ przekroju")

# Współczynniki sztywności z PDF [cite: 53, 54]
# a1 odpowiada za ściskanie/rozciąganie, a2 za zginanie
a1 = E * A / le
a2 = E * I / le**3

print(f"Dane: L={L}m, E={E:.2e}Pa, I={I:.2e}m^4, F={F_val}N")
print(f"Parametry elementu: a1={a1:.2e}, a2={a2:.2e}")

# --- 3. INICJALIZACJA STRUKTUR MES ---
n_nodes = n_elem + 1        # Liczba węzłów
dof_per_node = 3            # Stopnie swobody na węzeł (u, w, theta) [cite: 7]
total_dofs = n_nodes * dof_per_node

K_global = np.zeros((total_dofs, total_dofs)) # Globalna macierz sztywności [cite: 52]
F_vector = np.zeros(total_dofs)               # Wektor obciążeń

# --- 4. AGREGACJA MACIERZY SZTYWNOŚCI ---
# Definicja macierzy lokalnej k (6x6) na podstawie [cite: 38-51]
# Indeksy lokalne: 0:u1, 1:w1, 2:theta1, 3:u2, 4:w2, 5:theta2
k_local = np.array([
    [ a1,    0,       0,      -a1,    0,       0      ],
    [ 0,     12*a2,   6*a2*le, 0,    -12*a2,   6*a2*le],
    [ 0,     6*a2*le, 4*a2*le**2, 0, -6*a2*le, 2*a2*le**2],
    [-a1,    0,       0,       a1,    0,       0      ],
    [ 0,    -12*a2,  -6*a2*le, 0,     12*a2,  -6*a2*le],
    [ 0,     6*a2*le, 2*a2*le**2, 0, -6*a2*le, 4*a2*le**2]
])

for i in range(n_elem):
    # Indeksy w globalnej macierzy dla węzła i oraz i+1
    start_idx = i * dof_per_node
    end_idx = start_idx + 2 * dof_per_node # obejmuje 2 węzły (6 wierszy/kolumn)
    
    # Dodawanie macierzy lokalnej do globalnej (nakładanie się bloków)
    # Wykorzystujemy slicing numpy
    # Mapa indeksów globalnych dla elementu i:
    idx_map = [
        3*i, 3*i+1, 3*i+2,      # Węzeł i (u, w, theta)
        3*(i+1), 3*(i+1)+1, 3*(i+1)+2 # Węzeł i+1 (u, w, theta)
    ]
    
    for r in range(6):
        for c in range(6):
            K_global[idx_map[r], idx_map[c]] += k_local[r, c]

# --- 5. WARUNKI BRZEGOWE I OBCIĄŻENIA ---
# a) Siła przyłożona w ostatnim węźle
# Ostatni węzeł ma indeks (n_nodes - 1). 
# Z rysunku na str. 2  oś Z skierowana jest w DÓŁ. Siła F też w dół.
# Zatem siła jest dodatnia w kierunku osi Z.
node_last_idx = n_nodes - 1
idx_w_last = node_last_idx * 3 + 1 # DOF odpowiadający przemieszczeniu pionowemu w
F_vector[idx_w_last] = F_val

# b) Utwierdzenie w węźle 0 (u=0, w=0, theta=0) 
# Metoda usuwania wierszy i kolumn dla ustalonych stopni swobody (0, 1, 2)
fixed_dofs = [0, 1, 2]
free_dofs = [i for i in range(total_dofs) if i not in fixed_dofs]

# Zredukowany układ równań
K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
F_reduced = F_vector[free_dofs]

# --- 6. ROZWIĄZANIE ---
U_reduced = np.linalg.solve(K_reduced, F_reduced)

# Odtworzenie pełnego wektora przemieszczeń
U_full = np.zeros(total_dofs)
U_full[free_dofs] = U_reduced

# Wyciągnięcie przemieszczeń pionowych (w) dla każdego węzła
# w znajduje się na indeksach: 1, 4, 7, ...
displacements_w = U_full[1::3]
# Przeliczenie na mm do wykresu
displacements_w_mm = displacements_w * 1000.0

# Pozycje węzłów wzdłuż osi X
x_coords = np.linspace(0, L_mm, n_nodes)

# --- 7. ROZWIĄZANIE ANALITYCZNE ---
# Wzór: f = (F * x^2 * (3L - x)) / (6EI) - to ogólny wzór na ugięcie w punkcie x
# Ale w zadaniu [cite: 320] podano wzór na maksymalne ugięcie na końcu: f = (F * L^3) / (3EI)
# Policzmy ugięcie na końcu analitycznie dla porównania:
w_analytical_max = (F_val * L**3) / (3 * E * I)
w_analytical_max_mm = w_analytical_max * 1000.0

print(f"\nWyniki:")
print(f"Ugięcie MES na końcu: {displacements_w_mm[-1]:.4f} mm")
print(f"Ugięcie analityczne:  {w_analytical_max_mm:.4f} mm")

# --- 8. WYKRES [cite: 5] ---
plt.figure(figsize=(10, 6))
plt.plot(x_coords, displacements_w_mm, 'bo-', label='MES (Wynik numeryczny)')
# Rysowanie linii poziomej dla wartości analitycznej na końcu
plt.plot(L_mm, w_analytical_max_mm, 'rx', markersize=10, label='Rozwiązanie analityczne (koniec belki)')

plt.title(f'Wykres przemieszczeń belki (L={L_mm}mm, F={F_val}N)')
plt.xlabel('Długość belki x [mm]')
plt.ylabel('Przemieszczenie pionowe w(x) [mm]')
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis() # Odwracamy oś Y, bo ugięcie jest "w dół" (zgodnie z osią Z na rysunku)
plt.show()
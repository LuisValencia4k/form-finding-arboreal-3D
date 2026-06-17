"""
===============================================================================
MOTOR DE FORM-FINDING FEM 3D: PRINCIPIO DE FUNICULARIDAD ORTOGONAL
===============================================================================
Autor: Luis Alberto Valencia Pérez (Independent Researcher)
Fecha: Junio 2026
DOI de la Publicación: https://doi.org/10.5281/zenodo.20706928
Licencia: MIT License (https://opensource.org/licenses/MIT)

Descripción:
Este código implementa un motor de form-finding basado en el método de elementos finitos para estructuras con múltiples ramas, 
utilizando el principio de funicularidad ortogonal. El algoritmo optimiza la geometría de las ramas 
para minimizar el residuo global de fuerzas perpendiculares (T_res) en cada iteración, ajustando los ángulos de las ramas hijas 
y aplicando clipping geométrico para garantizar configuraciones físicamente viables. Se incluye un mecanismo de early stopping 
con rollback a la mejor configuración histórica para evitar estancamientos o divergencias. 
Además, se calcula detalladamente el cortante en la columna padre y se realiza un diagnóstico funicular basado 
en la energía de deformación axial vs flexión/corte. El código también ofrece herramientas para análisis espectral (kappa) 
y verificación estructural según MCFT, así como un post-proceso morfológico para dimensionamiento de fuste y ramas ahusadas.

Características principales:
- Modelado de elementos tipo viga de Timoshenko con rigidez a corte y torsión.
- Algoritmo de optimización con clipping geométrico para evitar configuraciones no físicas.
- Early stopping con rollback a la mejor configuración histórica para garantizar estabilidad.
- Cálculo detallado de fuerzas internas, ángulos de fuerza y cortantes en la columna padre.
- Diagnóstico funicular basado en energía de deformación axial vs flexión/corte.
- Análisis espectral (kappa) para evaluación de convergencia y estabilidad.
- Verificación estructural automatizada según MCFT (Modified Compression Field Theory).
- Post-proceso morfológico para dimensionamiento de fuste y ramas ahusadas.
===============================================================================
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

# ------------------------------------------------------------------------------
# CLASE PRINCIPAL
# ------------------------------------------------------------------------------
class FormFindingFEM:
    def __init__(self):
        self.nodes = []      
        self.elements = []   
        self.K_global = np.zeros((0, 0))
        self.F_global = np.zeros(0)
        self.u_global = np.zeros(0)
        self.n_dof = 0
        self.converged = False
        self.history_Tres = []
        self.history_Gvec = []
        self.history_Vcol = []
        self.history_alphas = []
        self.initial_node_coords = None   
        self.default_ks = 5.0/6.0   

    # --------------------------------------------------------------
    # 1. MALLADO Y CONDICIONES DE BORDE
    # --------------------------------------------------------------
    def add_node(self, x, y, z, fix=None, load=None):
        if fix is None:
            fix = [0,0,0,0,0,0]
        if load is None:
            load = [0.0,0.0,0.0,0.0,0.0,0.0]
        else:
            if len(load) == 3:
                load = [load[0], load[1], load[2], 0.0, 0.0, 0.0]
            elif len(load) != 6:
                raise ValueError("load debe tener 3 o 6 componentes")
        self.nodes.append({
            'x': float(x), 'y': float(y), 'z': float(z),
            'fix': fix,
            'load': np.array(load, dtype=float)
        })
        return len(self.nodes)-1

    def add_element(self, i, j, b, h, E, G, is_branch=False, is_trunk=False, is_rigid_link=False,
                    top_node=None, L_xy=None, alpha_inicial_deg=None):
        xi, yi, zi = self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']
        xj, yj, zj = self.nodes[j]['x'], self.nodes[j]['y'], self.nodes[j]['z']
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        L = math.sqrt(dx*dx + dy*dy + dz*dz)

        if is_branch:
            if top_node is None:
                raise ValueError("En ramas hijas debe indicarse 'top_node'.")
            node_sup = i if i == top_node else j
            node_inf = j if i == top_node else i
            
            xs, ys = self.nodes[node_sup]['x'], self.nodes[node_sup]['y']
            xi_inf, yi_inf = self.nodes[node_inf]['x'], self.nodes[node_inf]['y']
            if L_xy is None:
                L_xy = math.sqrt((xs - xi_inf)**2 + (ys - yi_inf)**2)
            if alpha_inicial_deg is None:
                dz_inicial = self.nodes[node_sup]['z'] - self.nodes[node_inf]['z']
                alpha_inicial_rad = math.atan2(dz_inicial, L_xy)
                alpha_inicial_deg = math.degrees(alpha_inicial_rad)
            else:
                alpha_inicial_rad = math.radians(alpha_inicial_deg)
                
            branch_params = {
                'node_sup': node_sup,
                'node_inf': node_inf,
                'L_xy': L_xy,
                'alpha_rad': alpha_inicial_rad,
                'alpha_deg': alpha_inicial_deg
            }
        else:
            branch_params = None

        A = b * h
        Iy, Iz = (h * b**3) / 12.0, (b * h**3) / 12.0
        a_max, b_min = max(b, h), min(b, h)
        J = (1/3.0) * a_max * b_min**3 * (1 - 0.63 * (b_min / a_max))
        Asy = Asz = A * self.default_ks

        element = {
            'i': i, 'j': j,
            'dx': dx, 'dy': dy, 'dz': dz, 'L': L,
            'b': b, 'h': h, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J,
            'Asy': Asy, 'Asz': Asz,
            'E': E, 'G': G,
            'is_branch': is_branch,
            'is_trunk': is_trunk,
            'is_rigid_link': is_rigid_link,
            'branch_params': branch_params
        }
        self.elements.append(element)
        return len(self.elements)-1

    # --------------------------------------------------------------
    # 2. MATRICES DE RIGIDEZ LOCAL Y TRANSFORMACIÓN
    # --------------------------------------------------------------
    def _k_local_3d_timoshenko(self, E, G, A, Iy, Iz, J, Asy, Asz, L):
        k = np.zeros((12,12))
        EA_L, GJ_L = E * A / L, G * J / L
        phi_y = 12 * E * Iz / (G * Asy * L**2)
        phi_z = 12 * E * Iy / (G * Asz * L**2)

        a_z = (12 * E * Iz) / (L**3 * (1 + phi_y))
        b_z = (6 * E * Iz)  / (L**2 * (1 + phi_y))
        c_z = (4 + phi_y) * E * Iz / (L * (1 + phi_y))
        d_z = (2 - phi_y) * E * Iz / (L * (1 + phi_y))
        
        a_y = (12 * E * Iy) / (L**3 * (1 + phi_z))
        b_y = (6 * E * Iy)  / (L**2 * (1 + phi_z))
        c_y = (4 + phi_z) * E * Iy / (L * (1 + phi_z))
        d_y = (2 - phi_z) * E * Iy / (L * (1 + phi_z))

        k[0,0] = k[6,6] = EA_L; k[0,6] = k[6,0] = -EA_L
        k[3,3] = k[9,9] = GJ_L; k[3,9] = k[9,3] = -GJ_L

        k[1,1] = k[7,7] = a_z
        k[1,7] = k[7,1] = -a_z
        k[1,5] = k[5,1] = b_z
        k[1,11] = k[11,1] = b_z
        k[5,7] = k[7,5] = -b_z
        k[7,11] = k[11,7] = -b_z
        k[5,5] = k[11,11] = c_z
        k[5,11] = k[11,5] = d_z

        k[2,2] = k[8,8] = a_y
        k[2,8] = k[8,2] = -a_y
        k[2,4] = k[4,2] = -b_y
        k[2,10] = k[10,2] = -b_y
        k[4,8] = k[8,4] = b_y
        k[8,10] = k[10,8] = b_y
        k[4,4] = k[10,10] = c_y
        k[4,10] = k[10,4] = d_y

        return k

    def _matriz_transformacion_3d(self, dx, dy, dz, L):
        cx, cy, cz = dx/L, dy/L, dz/L
        vec_x = np.array([cx, cy, cz])
        if abs(cz) > 0.9999:
            vec_y = np.array([0,1,0])
            vec_z = np.array([-1,0,0]) if cz > 0 else np.array([1,0,0])
        else:
            vec_Z_glob = np.array([0,0,1])
            vec_y = np.cross(vec_Z_glob, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
        lam = np.vstack([vec_x, vec_y, vec_z])
        T = np.zeros((12,12))
        for i in range(4):
            T[3*i:3*i+3, 3*i:3*i+3] = lam
        return T

    # --------------------------------------------------------------
    # 3. ENSAMBLAJE Y SOLUCIÓN DEL SISTEMA GLOBAL
    # --------------------------------------------------------------
    def _update_element_geometry(self, e):
        i, j = e['i'], e['j']
        e['dx'] = self.nodes[j]['x'] - self.nodes[i]['x']
        e['dy'] = self.nodes[j]['y'] - self.nodes[i]['y']
        e['dz'] = self.nodes[j]['z'] - self.nodes[i]['z']
        e['L'] = math.sqrt(e['dx']**2 + e['dy']**2 + e['dz']**2)

    def ensamblar_sistema_global(self):
        n_nodes = len(self.nodes)
        self.n_dof = 6 * n_nodes
        self.K_global = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof)

        for idx, node in enumerate(self.nodes):
            self.F_global[6*idx:6*idx+6] = node['load']

        for e in self.elements:
            self._update_element_geometry(e)
            T = self._matriz_transformacion_3d(e['dx'], e['dy'], e['dz'], e['L'])
            k_loc = self._k_local_3d_timoshenko(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'],
                                                e['Asy'], e['Asz'], e['L'])
            K_elem = T.T @ k_loc @ T
            dofs = [6*e['i'] + k for k in range(6)] + [6*e['j'] + k for k in range(6)]
            for r in range(12):
                for c in range(12):
                    self.K_global[dofs[r], dofs[c]] += K_elem[r, c]

        for idx, node in enumerate(self.nodes):
            dof0 = 6*idx
            for d in range(6):
                if node['fix'][d] == 1:
                    self.K_global[dof0+d, :] = 0
                    self.K_global[:, dof0+d] = 0
                    self.K_global[dof0+d, dof0+d] = 1.0
                    self.F_global[dof0+d] = 0.0

    def resolver(self):
        try:
            self.u_global = np.linalg.solve(self.K_global, self.F_global)
        except np.linalg.LinAlgError:
            self.u_global = np.linalg.lstsq(self.K_global, self.F_global, rcond=None)[0]

    # --------------------------------------------------------------
    # 4. CÁLCULO DEL RESIDUO ORTOGONAL GLOBAL
    # --------------------------------------------------------------
    def obtener_fuerzas_nodales_elemento(self, e):
        self._update_element_geometry(e)
        u_elem = np.concatenate([self.u_global[6*e['i']:6*e['i']+6],
                                 self.u_global[6*e['j']:6*e['j']+6]])
        T = self._matriz_transformacion_3d(e['dx'], e['dy'], e['dz'], e['L'])
        k_loc = self._k_local_3d_timoshenko(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'],
                                            e['Asy'], e['Asz'], e['L'])
        return T.T @ (k_loc @ (T @ u_elem))

    def compute_T_res_global(self):
        return np.linalg.norm(self.compute_T_res_vec())

    def compute_T_res_vec(self):
        T_vec = np.zeros(3)
        for e in self.elements:
            if not e.get('is_branch', False):
                continue
            f_glob = self.obtener_fuerzas_nodales_elemento(e)
            bp = e['branch_params']
            if bp['node_inf'] == e['i']:
                F = f_glob[0:3]
                u_hat = np.array([e['dx'], e['dy'], e['dz']]) / e['L']
            else:
                F = f_glob[6:9]
                u_hat = -np.array([e['dx'], e['dy'], e['dz']]) / e['L']
            F_paralelo = np.dot(F, u_hat) * u_hat
            T_vec += F - F_paralelo
        return T_vec
    

    # --------------------------------------------------------------
    # 5. BUCLE PRINCIPAL DE FORM-FINDING (CON CLIPPING, ROLLBACK Y LOGS)
    # --------------------------------------------------------------
    def form_finding_loop(self, max_iter=10000, tol=1e-2, w_max=0.3, w_min=0.01, lam=0.1, patience=100, verbose=True):
        self.initial_node_coords = deepcopy([{'x':n['x'], 'y':n['y'], 'z':n['z']} for n in self.nodes])
        
        self.history_Tres = []      
        self.history_Gvec = []      
        self.history_Vcol = [] 
        self.history_alphas = []
        current_alphas = [e['branch_params']['alpha_rad'] for e in self.elements if e.get('is_branch', False)]
        self.history_alphas.append(current_alphas)     

        # --- VARIABLES PARA EL ROLLBACK (INICIALIZADAS FUERA DEL BUCLE) ---
        best_Tres = float('inf')
        best_iter = 0
        best_nodes = deepcopy([{'x':n['x'], 'y':n['y'], 'z':n['z']} for n in self.nodes])
        best_branches = [e['branch_params']['alpha_rad'] for e in self.elements if e.get('is_branch', False)]
        patience_counter = 0

        # --- BUCLE PRINCIPAL DE OPTIMIZACIÓN ---
        for it in range(max_iter):
            self.ensamblar_sistema_global()
            self.resolver()
            
            # Cálculo del residuo global y fuerzas de corte en columna padre
            G_vec = self.compute_T_res_vec()
            T_res = np.linalg.norm(G_vec)
            
            # Cálculo del cortante en la columna padre (asumiendo que es el primer elemento marcado como 'is_trunk')
            V_col = 0.0
            for e in self.elements:
                if e.get('is_trunk', False):
                    f_glob = self.obtener_fuerzas_nodales_elemento(e)
                    F_tope = f_glob[6:9]
                    V_col = math.sqrt(F_tope[0]**2 + F_tope[1]**2)
                    break
            
            # Guardar históricos para análisis y gráficos
            self.history_Tres.append(T_res)
            self.history_Gvec.append(G_vec.copy())
            self.history_Vcol.append(V_col)
            self.history_alphas.append(current_alphas)
            
            # --- LÓGICA DE EARLY STOPPING Y ROLLBACK ---
            if T_res < best_Tres:
                best_Tres = T_res
                best_iter = it
                best_nodes = deepcopy([{'x':n['x'], 'y':n['y'], 'z':n['z']} for n in self.nodes])
                best_branches = [e['branch_params']['alpha_rad'] for e in self.elements if e.get('is_branch', False)]
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Criterio de parada: estancamiento o divergencia
            if patience_counter > patience or math.isnan(T_res) or T_res > best_Tres * 100:
                if verbose:
                    print(f"\n[Early Stopping] Estancamiento detectado tras {patience} iteraciones sin mejora.")
                    print(f"[Rollback] Restaurando mínimo histórico (iter {best_iter}, T_res = {best_Tres:.4f} kN)")
                
                # Restaurar geometría óptima
                for idx, n in enumerate(self.nodes):
                    n['x'] = best_nodes[idx]['x']
                    n['y'] = best_nodes[idx]['y']
                    n['z'] = best_nodes[idx]['z']
                
                b_idx = 0
                for e in self.elements:
                    if e.get('is_branch', False):
                        bp = e['branch_params']
                        bp['alpha_rad'] = best_branches[b_idx]
                        bp['alpha_deg'] = math.degrees(best_branches[b_idx])
                        b_idx += 1
                        
                # Podar historiales para que las gráficas usen el punto óptimo
                self.history_Tres = self.history_Tres[:best_iter+1]
                self.history_Gvec = self.history_Gvec[:best_iter+1]
                self.history_Vcol = self.history_Vcol[:best_iter+1]
                break

            # --- REPORTE DETALLADO ---
            if verbose:
                print(f"\nIter {it}: T_res = {T_res:.4e} kN, V_fuste = {V_col:.4f} kN")
                print(f"           G = ({G_vec[0]:8.4f}, {G_vec[1]:8.4f}, {G_vec[2]:8.4f}) kN")
                print(f"  Columna Padre -> Cortante absorbido (Hiperestaticidad): {V_col:.4f} kN")

            # Criterio de convergencia basado en el residuo global
            if T_res < tol:
                if verbose:
                    print(f"\nConvergencia lograda en iteración {it}! (T_res = {T_res:.3e} kN)")
                self.converged = True
                break
            
            # --- ACTUALIZACIÓN GEOMÉTRICA DE RAMAS ---
            for e in self.elements:
                if e.get('is_branch', False):
                    bp = e['branch_params']
                    f_glob = self.obtener_fuerzas_nodales_elemento(e)
                    F = f_glob[0:3] if bp['node_inf'] == e['i'] else f_glob[6:9]
                    F_horiz = math.sqrt(F[0]**2 + F[1]**2)
                    
                    # Cálculo del ángulo de fuerza (theta_F) y comparación con el ángulo actual (alpha_old)
                    theta_F = math.atan2(abs(F[2]), F_horiz)
                    alpha_old = bp['alpha_rad']
                    
                    # Cálculo y reporte de fuerza perpendicular local
                    u_hat = np.array([e['dx'], e['dy'], e['dz']]) / e['L']
                    if bp['node_inf'] != e['i']:
                        u_hat = -u_hat
                    F_paralelo = np.dot(F, u_hat) * u_hat
                    F_perp = F - F_paralelo
                    F_perp_mag = np.linalg.norm(F_perp)
                    
                    # Reporte detallado para cada rama
                    if verbose:
                        print(f"  Rama {bp['node_sup']} -> theta_F: {math.degrees(theta_F):.4f}° | alpha_old: {math.degrees(alpha_old):.4f}° | F_perp local: {F_perp_mag:.4f} kN")
                    
                    w_i = w_max * math.exp(-lam * it) + w_min
                    alpha_new_teorico = alpha_old + w_i * (theta_F - alpha_old)
                    
                    # Clipping geométrico
                    Z_sup = self.nodes[bp['node_sup']]['z']
                    Z_min = self.nodes[1]['z'] 
                    Z_max = Z_sup - 0.05 
                    
                    # Cálculo de la nueva posición z de la rama hija y actualización del ángulo real basado en el desplazamiento permitido
                    Z_teorico = Z_sup - bp['L_xy'] * math.tan(alpha_new_teorico)
                    Z_real = float(np.clip(Z_teorico, Z_min, Z_max))
                    alpha_real_rad = math.atan2(Z_sup - Z_real, bp['L_xy'])
                    
                    # Actualización de la posición z del nodo inferior de la rama y del ángulo en los parámetros de la rama
                    self.nodes[bp['node_inf']]['z'] = Z_real
                    bp['alpha_rad'] = alpha_real_rad
                    bp['alpha_deg'] = math.degrees(alpha_real_rad)

        return self.history_Tres, self.history_Gvec, self.history_Vcol

    # --------------------------------------------------------------
    # POST-PROCESO: VERIFICACIÓN MCFT, ENERGÍA, MORFOLOGÍA
    # --------------------------------------------------------------
    def post_process_MCFT(self, T_res_kN, f_c_MPa, E_s_MPa, A_s_mm2, A_biela_mm2, V_u_kN, theta_deg):
        T_res_N = T_res_kN * 1000.0
        eps_t = T_res_N / (E_s_MPa * A_s_mm2)
        beta_s = (E_s_MPa * A_s_mm2) / (0.8 * E_s_MPa * A_s_mm2 + 170.0 * T_res_N)
        beta_s = min(beta_s, 1.0)
        F_nn_N = 0.85 * f_c_MPa * A_biela_mm2 * beta_s
        F_nn_kN = F_nn_N / 1000.0
        theta_rad = math.radians(theta_deg)
        capacidad = F_nn_kN * math.sin(theta_rad)
        margen = capacidad / V_u_kN if V_u_kN > 0 else 0
        status = "CUMPLE ✓" if capacidad >= V_u_kN else "NO CUMPLE ✗"
        print("\n=== POST-PROCESO: VERIFICACIÓN MCFT ===")
        print(f"Demanda neta de hendimiento (|T_res|) : {T_res_kN:.2f} kN")
        print(f"Deformación transversal (ε_t)         : {eps_t:.6f} mm/mm")
        print(f"Factor de eficiencia (β_s)            : {beta_s:.3f}")
        print(f"Capacidad nominal del puntal (F_nn)   : {F_nn_kN:.2f} kN")
        print(f"Capacidad vertical disponible         : {capacidad:.2f} kN")
        print(f"Cortante último actuante (V_u)        : {V_u_kN:.2f} kN")
        print(f"Ratio Demanda/Capacidad               : {1/margen:.2f}" if margen > 0 else "N/A")
        print(f"Estado de Verificación                : {status}")
        return beta_s, capacidad

    def diagnostico_funicular(self):
        U_axial = 0.0
        U_flex = 0.0

        # Cálculo de energía de deformación axial y flexión/corte para cada elemento
        for e in self.elements:
            if e.get('is_rigid_link', False):
                continue
            # Actualizar geometría del elemento para asegurar que dx, dy, dz y L estén correctos
            self._update_element_geometry(e)
            i, j = e['i'], e['j']
            u_elem_global = np.concatenate([self.u_global[6*i:6*i+6], self.u_global[6*j:6*j+6]])
            T = self._matriz_transformacion_3d(e['dx'], e['dy'], e['dz'], e['L'])
            k_loc = self._k_local_3d_timoshenko(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'],
                                                e['Asy'], e['Asz'], e['L'])
            u_loc = T @ u_elem_global
            f_loc = k_loc @ u_loc
            u_ax = 0.5 * (f_loc[0]*u_loc[0] + f_loc[6]*u_loc[6])
            u_fl = 0.5 * np.sum(f_loc[1:6] * u_loc[1:6]) + 0.5 * np.sum(f_loc[7:12] * u_loc[7:12])
            U_axial += u_ax
            U_flex += u_fl
        U_total = U_axial + U_flex
        if U_total == 0:
            return
        
        # Cálculo de porcentajes y clasificación
        pct_axial = (U_axial / U_total) * 100
        pct_flex = (U_flex / U_total) * 100
        indice = U_axial / U_total
        if U_total < 1e-3:
            energia_str = f"{U_total * 1e6:.2f} µJ"
        elif U_total < 1.0:
            energia_str = f"{U_total * 1000:.2f} mJ"
        else:
            energia_str = f"{U_total:.2f} J"
        clasif = ("ESTRUCTURA FUNICULAR" if indice >= 0.95 else
                  "CUASI FUNICULAR" if indice >= 0.85 else
                  "SISTEMA HÍBRIDO (FLEXO-COMPRESIÓN)" if indice >= 0.60 else
                  "ESTRUCTURA A FLEXIÓN DOMINANTE")
        print("\n================================================")
        print("            DIAGNÓSTICO FUNICULAR               ")
        print("================================================")
        print(f"T_res global remanente = {self.history_Tres[-1]:.2f} kN")
        print(f"Energía de deformación = {energia_str} (Bruto: {U_total:.4e} J)")
        print("------------------------------------------------")
        print(f"Energía axial          = {pct_axial:.1f} %")
        print(f"Energía flexión/corte  = {pct_flex:.1f} %")
        print(f"\nÍndice funicular       = {indice:.3f}")
        print(f"\nClasificación: [{pct_axial:.1f}%] {clasif}")
        print("================================================")

    def optimizar_dimensiones_circulares(self, f_c_MPa=30.0, h_losa_m=0.25, E_concreto_Pa=30e9):
        print("\n================================================")
        print("  OPTIMIZACIÓN MORFOLÓGICA (CÍRCULOS AHUSADOS)  ")
        print("================================================")
        d_losa_efectivo = h_losa_m - 0.04
        phi_punz, phi_comp = 0.75, 0.65
        v_c_Pa = 0.33 * math.sqrt(f_c_MPa) * 1e6
        f_c_Pa = f_c_MPa * 1e6
        D_padre = 0.10
        d_hija_tope = 0.10
        D_hija_base = 0.10
        step = 0.05
        Vu_max_punz = max((abs(node['load'][2]) for node in self.nodes if node['load'][2] < 0), default=0.0)
        Vu_max_N = Vu_max_punz * 1000.0
        while True:
            b_o = math.pi * (d_hija_tope + d_losa_efectivo)
            V_resistido = phi_punz * v_c_Pa * b_o * d_losa_efectivo
            if V_resistido >= Vu_max_N:
                break
            d_hija_tope += step
        D_hija_base = d_hija_tope
        Pu_rama_max = 0.0
        L_rama_max = 0.0
        for e in self.elements:
            if e.get('is_branch', False):
                f_glob = self.obtener_fuerzas_nodales_elemento(e)
                F_axial = np.linalg.norm(f_glob[0:3])
                Pu_rama_max = max(Pu_rama_max, F_axial)
                L_rama_max = max(L_rama_max, e['L'])
        Pu_rama_N = Pu_rama_max * 1000.0
        while True:
            d_mid = (D_hija_base + d_hija_tope) / 2.0
            I_avg = (math.pi * d_mid**4) / 64.0
            P_cr = (math.pi**2 * E_concreto_Pa * I_avg) / (L_rama_max**2)
            if 0.75 * P_cr >= Pu_rama_N:
                break
            D_hija_base += step
        D_padre = max(D_padre, D_hija_base)
        Pu_padre_N = 0.0
        Mu_padre_Nm = 0.0
        for e in self.elements:
            if e.get('is_trunk', False):
                f_glob = self.obtener_fuerzas_nodales_elemento(e)
                Pu_padre_N = abs(f_glob[2]) * 1000.0
                V_hiper_N = math.sqrt(f_glob[6]**2 + f_glob[7]**2) * 1000.0
                Mu_padre_Nm = V_hiper_N * e['L']
                break
        while True:
            Area = (math.pi * D_padre**2) / 4.0
            S_modulo = (math.pi * D_padre**3) / 32.0
            esfuerzo_max = (Pu_padre_N / Area) + (Mu_padre_Nm / S_modulo)
            esfuerzo_resistente = phi_comp * 0.85 * f_c_Pa
            cond_geom = D_padre >= (1.5 * D_hija_base)
            if esfuerzo_max <= esfuerzo_resistente and cond_geom:
                break
            D_padre += step
        print("         RESULTADOS DEL DIMENSIONAMIENTO        ")
        print("------------------------------------------------")
        print(f"Fuste Padre (D)        : {D_padre*100:.1f} cm   [Gobierna: Flexocompresión & Geometría]")
        print(f"Rama Hija Base (D_inf) : {D_hija_base*100:.1f} cm   [Gobierna: Pandeo de Euler]")
        print(f"Rama Hija Tope (d_sup) : {d_hija_tope*100:.1f} cm   [Gobierna: Punzonamiento Losa]")
        print(f"Ratio de ahusamiento de rama : {d_hija_tope/D_hija_base:.2f}")
        return D_padre, D_hija_base, d_hija_tope

    # --------------------------------------------------------------
    # VISUALIZACIÓN Y MAPAS TOPOGRÁFICOS
    # --------------------------------------------------------------
    def plot_results(self, show_initial=True, show_plot=True):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.semilogy(self.history_Tres, 'o-', color='#2b6cb0', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('T_res (kN)')
        plt.title('Minimización del Residuo Nodal (Hendimiento)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        ax = plt.subplot(1,2,2, projection='3d')
        for e in self.elements:
            i, j = e['i'], e['j']
            xi, yi, zi = self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']
            xj, yj, zj = self.nodes[j]['x'], self.nodes[j]['y'], self.nodes[j]['z']
            if e.get('is_branch', False):
                color, lbl = 'blue', 'Rama hija'
            elif e.get('is_rigid_link', False):
                color, lbl = 'black', 'Capitel (Rígido)'
            else:
                color, lbl = 'red', 'Columna padre'
            ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, linewidth=2, label=lbl)
        for node in self.nodes:
            ax.scatter(node['x'], node['y'], zs=node['z'], c='black', s=30, depthshade=True)
        if show_initial and self.initial_node_coords:
            for e in self.elements:
                if e.get('is_branch', False):
                    i, j = e['i'], e['j']
                    xi0 = self.initial_node_coords[i]['x']
                    yi0 = self.initial_node_coords[i]['y']
                    zi0 = self.initial_node_coords[i]['z']
                    xj0 = self.initial_node_coords[j]['x']
                    yj0 = self.initial_node_coords[j]['y']
                    zj0 = self.initial_node_coords[j]['z']
                    ax.plot([xi0, xj0], [yi0, yj0], [zi0, zj0], 'gray', linestyle=':', linewidth=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)') #type: ignore
        ax.set_title('Morfología Arbórea Final')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        if show_plot:
            plt.show()

    def plot_gradient_field_2d_n_ramas(self, b_idx1=0, b_idx2=1, resolution=25, margin_deg=10.0, show_plot=True):
        ramas = [e for e in self.elements if e.get('is_branch', False)]
        if len(ramas) < 2:
            return
            
        r1 = ramas[b_idx1]
        r2 = ramas[b_idx2]
        
        node_sup_1 = r1['branch_params']['node_sup']
        node_sup_2 = r2['branch_params']['node_sup']
        print(f"  -> Proyección transversal 2D: Rama {node_sup_1} vs Rama {node_sup_2}...")
        
        a1_final = r1['branch_params']['alpha_rad']
        a2_final = r2['branch_params']['alpha_rad']
        margin_rad = math.radians(margin_deg)
        
        a1_vals = np.linspace(a1_final - margin_rad, a1_final + margin_rad, resolution)
        a2_vals = np.linspace(a2_final - margin_rad, a2_final + margin_rad, resolution)
        A1, A2 = np.meshgrid(a1_vals, a2_vals)
        Z_Tres = np.zeros_like(A1)
        
        orig_nodes = deepcopy(self.nodes)
        orig_alphas = [r['branch_params']['alpha_rad'] for r in ramas]
        
        for i in range(resolution):
            for j in range(resolution):
                r1['branch_params']['alpha_rad'] = A1[i, j]
                r2['branch_params']['alpha_rad'] = A2[i, j]
                for r in [r1, r2]:
                    bp = r['branch_params']
                    Z_sup = self.nodes[bp['node_sup']]['z']
                    self.nodes[bp['node_inf']]['z'] = Z_sup - bp['L_xy'] * math.tan(bp['alpha_rad'])
                self.ensamblar_sistema_global()
                self.resolver()
                Z_Tres[i, j] = self.compute_T_res_global()
                
        self.nodes = deepcopy(orig_nodes)
        for idx, r in enumerate(ramas):
            r['branch_params']['alpha_rad'] = orig_alphas[idx]
        self.ensamblar_sistema_global()
        self.resolver()
        
        dZ_dA2, dZ_dA1 = np.gradient(Z_Tres, a2_vals, a1_vals)
        U, V = -dZ_dA1, -dZ_dA2
        M = np.hypot(U, V)
        M[M == 0] = 1.0
        Un, Vn = U/M, V/M
        
        plt.figure(figsize=(10,8))
        plt.contourf(np.degrees(A1), np.degrees(A2), Z_Tres, levels=40, cmap='viridis', alpha=0.85)
        plt.colorbar(label='Magnitud del Residuo Nodal $T_{res}$ (kN)')
        plt.contour(np.degrees(A1), np.degrees(A2), Z_Tres, levels=20, colors='black', linewidths=0.5, alpha=0.6)
        plt.quiver(np.degrees(A1), np.degrees(A2), Un, Vn, M, cmap='autumn', pivot='mid', scale=35, width=0.003, alpha=0.9)
        
        if hasattr(self, 'history_alphas') and len(self.history_alphas) > 0:
            hist_a1 = [math.degrees(a[b_idx1]) for a in self.history_alphas]
            hist_a2 = [math.degrees(a[b_idx2]) for a in self.history_alphas]
            plt.plot(hist_a1, hist_a2, color='white', linestyle='--', linewidth=2, label='Trayectoria Dinámica')
            plt.plot(hist_a1[0], hist_a2[0], 'wo', markeredgecolor='black', markersize=8, label='Arranque Inicial')
            
        plt.plot(math.degrees(a1_final), math.degrees(a2_final), 'r*', markersize=18, markeredgecolor='black', label='Óptimo Alcanzado')
        
        min_idx = np.unravel_index(np.argmin(Z_Tres, axis=None), Z_Tres.shape)
        a1_min = math.degrees(a1_vals[min_idx[1]])
        a2_min = math.degrees(a2_vals[min_idx[0]])
        plt.plot(a1_min, a2_min, 'kX', markersize=12, label='Mínimo Teórico Local')
        
        plt.xlabel(f'Ángulo $\\alpha_{{{node_sup_1}}}$ de la Rama {node_sup_1} (grados)')
        plt.ylabel(f'Ángulo $\\alpha_{{{node_sup_2}}}$ de la Rama {node_sup_2} (grados)')
        plt.title(f'Topografía del Residuo Ortogonal (Rama {node_sup_1} vs Rama {node_sup_2})')
        plt.xlim([math.degrees(a1_vals[0]), math.degrees(a1_vals[-1])])
        plt.ylim([math.degrees(a2_vals[0]), math.degrees(a2_vals[-1])])
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        if show_plot:
            plt.show()

    def generate_all_plots(self, resolution=20, margin_deg=10.0):
        print("\n================================================")
        print("      GENERANDO VISUALIZACIONES MÚLTIPLES       ")
        print("================================================")
        # 1. Ventana Principal: Historial y Morfología 3D
        print("  -> Renderizando vista 3D e historial de convergencia...")
        self.plot_results(show_initial=True, show_plot=False)
        
        # 2. Ventanas Secundarias: Radiografías 2D del campo gradiente
        ramas = [e for e in self.elements if e.get('is_branch', False)]
        n = len(ramas)
        if n >= 2:
            import itertools
            # Combinamos todos los pares posibles de ramas (Ej: 3 ramas = 3 gráficas)
            pairs = list(itertools.combinations(range(n), 2))
            
            # Cinturón de seguridad: Si le metes un bosque de 10 ramas, evitamos que la PC explote
            if len(pairs) > 10:
                print(f"  [Aviso] Se detectaron {n} ramas. Para evitar sobrecarga gráfica, solo se procesarán 10 proyecciones.")
                pairs = pairs[:10] 
            
            # Renderizamos cada par en paralelo (sin mostrar aún)
            for idx1, idx2 in pairs:
                self.plot_gradient_field_2d_n_ramas(b_idx1=idx1, b_idx2=idx2, resolution=resolution, margin_deg=margin_deg, show_plot=False)
        
        # 3. Mostrar TODAS las ventanas generadas en paralelo
        print("  -> ¡Desplegando ventanas!")
        plt.show()


# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES PARA CONSTRUCCIÓN Y EXPERIMENTOS (GENERALIZADAS)
# ------------------------------------------------------------------------------
def build_arboreal_structure_general(base_fix, branch_data_list, capitel_rigido=True):
    solver = FormFindingFEM()
    solver.add_node(0.0, 0.0, 0.0, fix=base_fix)
    solver.add_node(0.0, 0.0, 1.5)
    
    arranques = []
    tops = []
    
    for br in branch_data_list:
        xg, yg = br['ground_coords']
        xt, yt, zt = br['top_coords']
        load = br.get('load', (0,0,-300.0,0,0,0))
        
        # Fijeza horizontal en la losa [1,1,0,1,1,1]
        top = solver.add_node(xt, yt, zt, fix=[1,1,0,1,1,1], load=load)
        tops.append(top)
        
        # Cálculo real del arranque Z inicial según el ángulo deseado
        L_xy = math.sqrt((xt - xg)**2 + (yt - yg)**2)
        alpha_rad = math.radians(br['alpha_deg'])
        z_inf_inicial = zt - L_xy * math.tan(alpha_rad)
        
        # Agregar nodo de arranque con coordenada Z ajustada para que la rama comience con el ángulo deseado
        inf = solver.add_node(xg, yg, z_inf_inicial)
        arranques.append(inf)

    # Elemento vertical rígido del fuste padre    
    solver.add_element(0, 1, b=0.4, h=0.4, E=30e9, G=12e9, is_trunk=True)
    E_cap = 30e12 if capitel_rigido else 30e9
    G_cap = 12e12 if capitel_rigido else 12e9
    
    # Elementos rígidos de capitel (si se activa)
    for inf in arranques:
        solver.add_element(1, inf, b=1.0, h=1.0, E=E_cap, G=G_cap, is_rigid_link=True)

    # Elementos de ramas hijas (con parámetros de flexibilidad y ángulo inicial)    
    for inf, top, br in zip(arranques, tops, branch_data_list):
        solver.add_element(inf, top, b=0.3, h=0.3, E=30e9, G=12e9,
                           is_branch=True, top_node=top, alpha_inicial_deg=br['alpha_deg'])
    return solver

def get_hiper_reactions(solver):
    solver.ensamblar_sistema_global()
    solver.resolver()
    for e in solver.elements:
        if e.get('is_trunk', False):
            f_glob = solver.obtener_fuerzas_nodales_elemento(e)
            F_tope = f_glob[6:9]
            M_tope = f_glob[9:12]
            V = np.sqrt(F_tope[0]**2 + F_tope[1]**2) / 1000.0
            M = np.sqrt(M_tope[0]**2 + M_tope[1]**2) / 1000.0
            return V, M
    return 0.0, 0.0

def compute_jacobian_alpha(solver, delta=1e-4):
    """
    Jacobiano numérico de G respecto a los ángulos de todas las ramas.
    Válido para n ramas arbitrarias. Retorna J (3×n), G0_vec, grad (J^T G), norm_grad.
    """
    branch_elems = [e for e in solver.elements if e.get('is_branch', False)]
    n = len(branch_elems)
    if n == 0:
        return None, None, None, None

    # Estado base
    solver.ensamblar_sistema_global()
    solver.resolver()
    G0_vec = solver.compute_T_res_vec()

    J = np.zeros((3, n))

    for k, e in enumerate(branch_elems):
        bp = e['branch_params']

        # Guardar estado completo
        alpha_orig = bp['alpha_rad']
        node_inf   = bp['node_inf']
        z_orig     = solver.nodes[node_inf]['z']

        # Perturbación +delta
        alpha_p = alpha_orig + delta
        bp['alpha_rad'] = alpha_p
        bp['alpha_deg'] = math.degrees(alpha_p)
        solver.nodes[node_inf]['z'] = (
            solver.nodes[bp['node_sup']]['z'] - bp['L_xy'] * math.tan(alpha_p)
        )
        solver.ensamblar_sistema_global()
        solver.resolver()
        Gp = solver.compute_T_res_vec()

        # Perturbación -delta
        alpha_m = alpha_orig - delta
        bp['alpha_rad'] = alpha_m
        bp['alpha_deg'] = math.degrees(alpha_m)
        solver.nodes[node_inf]['z'] = (
            solver.nodes[bp['node_sup']]['z'] - bp['L_xy'] * math.tan(alpha_m)
        )
        solver.ensamblar_sistema_global()
        solver.resolver()
        Gm = solver.compute_T_res_vec()

        # Diferencia centrada
        J[:, k] = (Gp - Gm) / (2.0 * delta)

        # Restaurar
        bp['alpha_rad'] = alpha_orig
        bp['alpha_deg'] = math.degrees(alpha_orig)
        solver.nodes[node_inf]['z'] = z_orig

    # Dejar solver en estado original
    solver.ensamblar_sistema_global()
    solver.resolver()

    grad     = J.T @ G0_vec          # ∇Φ = J^T G  (n-vector)
    norm_grad = float(np.linalg.norm(grad))
    return J, G0_vec, grad, norm_grad

def run_experiments():
    OPT_PARAMS = {"max_iter": 10000, "tol": 1e-2, "w_max": 0.1, "w_min": 0.01, "lam": 0.1, "verbose": False}
    print("\n" + "="*70)
    print("EXPERIMENTO A: CORRELACIÓN CON EL GRADO DE HIPERESTATICIDAD")
    print("="*70)
    branch_data_2 = [
        {'top_coords': (2.0, 0.0, 5.0), 'ground_coords': (0.0, 0.1), 'alpha_deg': 35.0, 'load': (0,0,-300.0,0,0,0)},
        {'top_coords': (-1.5, 1.2, 5.0), 'ground_coords': (0.0, -0.1), 'alpha_deg': 35.0, 'load': (0,0,-300.0,0,0,0)}
    ]
    casos = [
        ("Caso 1: Base articulada, sin capitel rígido", [1,1,1,0,0,0], False),
        ("Caso 2: Base articulada, con capitel rígido", [1,1,1,0,0,0], True),
        ("Caso 3: Base empotrada, capitel rígido",      [1,1,1,1,1,1], True),
    ]
    for desc, fix, cap_rig in casos:
        s0 = build_arboreal_structure_general(fix, branch_data_2, capitel_rigido=cap_rig)
        V0, M0 = get_hiper_reactions(s0)
        T0 = s0.compute_T_res_global()
        s1 = build_arboreal_structure_general(fix, branch_data_2, capitel_rigido=cap_rig)
        s1.form_finding_loop(**OPT_PARAMS)
        Vf, Mf = get_hiper_reactions(s1)
        Tf = s1.history_Tres[-1] if s1.history_Tres else float('nan')
        print(f"\n{desc}")
        print(f"  Antes  : V={V0:.2f} kN  M={M0:.2f} kN·m  T_res={T0:.2f} kN")
        print(f"  Después: V={Vf:.2f} kN  M={Mf:.2f} kN·m  T_res={Tf:.2f} kN")
        red_V = (V0 - Vf) / V0 * 100 if V0 > 0 else 0.0
        red_T = (T0 - Tf) / T0 * 100 if T0 > 0 else 0.0
        print(f"  Reducción: ΔV={red_V:.1f}%  ΔT_res={red_T:.1f}%")
        
    print("\n" + "="*70)
    print("EXPERIMENTO B: LIBERACIÓN PROGRESIVA DE GRADOS DE LIBERTAD")
    print("="*70)
    configs = [
        ("Empotrado (todos fijos)",               [1,1,1,1,1,1]),
        ("Liberar rotación X",                    [1,1,1,0,1,1]),
        ("Liberar rotaciones X y Y",              [1,1,1,0,0,1]),
        ("Liberar rotaciones X,Y,Z (articulado)", [1,1,1,0,0,0]),
        ("Liberar también traslación X",          [0,1,1,0,0,0]),
        ("Liberar también traslación Y",          [0,0,1,0,0,0]),
    ]
    for desc, fix in configs:
        s0 = build_arboreal_structure_general(fix, branch_data_2, capitel_rigido=True)
        V0, M0 = get_hiper_reactions(s0)
        T0 = s0.compute_T_res_global()
        s1 = build_arboreal_structure_general(fix, branch_data_2, capitel_rigido=True)
        s1.form_finding_loop(**OPT_PARAMS)
        Vf, Mf = get_hiper_reactions(s1)
        Tf = s1.history_Tres[-1] if s1.history_Tres else float('nan')
        print(f"\n{desc}")
        print(f"  Antes  : V={V0:.2f} kN  M={M0:.2f} kN·m  T_res={T0:.2f} kN")
        print(f"  Después: V={Vf:.2f} kN  M={Mf:.2f} kN·m  T_res={Tf:.2f} kN")
        red_V = (V0 - Vf) / V0 * 100 if V0 > 0 else 0.0
        red_T = (T0 - Tf) / T0 * 100 if T0 > 0 else 0.0
        print(f"  Reducción: ΔV={red_V:.1f}%  ΔT_res={red_T:.1f}%")

# ------------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    RUN_EXPERIMENTS = False  # Cambia a False para ejecutar solo el caso de estudio definitivo  

    if RUN_EXPERIMENTS:
        run_experiments()
    else:
        # RAMAS: Coordenadas, ángulo inicial y cargas aplicadas (generalizado para n ramas)
        # Añadir o quitar diccionarios en la lista para modificar el número de ramas sin tocar la lógica del código
        branch_data = [
            {'top_coords': (2.8, 1.0, 6.0), 'ground_coords': (0.15, 0.0), 'alpha_deg': 35.0, 'load': (0,0,-600.0, 0,0,0)},
            {'top_coords': (-1.5, 2.2, 5.0), 'ground_coords': (-0.05, 0.1), 'alpha_deg': 35.0, 'load': (0,0,-350.0, 0,0,0)},
            {'top_coords': (-1.0, -2.5, 4.2), 'ground_coords': (-0.05, -0.1), 'alpha_deg': 35.0, 'load': (0,0,-200.0, 0,0,0)},
            {'top_coords': (0.5, 1.8, 4.5), 'ground_coords': (0.05, 0.10), 'alpha_deg': 35.0, 'load': (0,0,-150.0, 0,0,0)} # Eliminar para probar con 3 ramas
            # Agrega más ramas aquí para experimentar con 5, 6 o más ramas sin necesidad de cambiar nada en la lógica del código. 
            # Solo asegúrate de que cada rama tenga su 'top_coords', 'ground_coords', 'alpha_deg' y 'load' definidos correctamente.
        ]

        # Construimos el solver con la configuración deseada (base empotrada, capitel rígido, n ramas según la lista)
        solver = build_arboreal_structure_general([1,1,1,1,1,1], branch_data, capitel_rigido=True)
        
        # Ejecutamos la optimización (el Rollback y Early Stopping están activos)
        solver.form_finding_loop(max_iter=10000, tol=1e-2, verbose=True)
        
        print("\n--- RESULTADOS FINALES ---")
        alpha_critico = 90.0
        for e in solver.elements:
            if e.get('is_branch', False):
                bp = e['branch_params']
                print(f"Rama {bp['node_sup']}: α = {bp['alpha_deg']:.2f}°, Z_inf = {solver.nodes[bp['node_inf']]['z']:.4f} m")
                alpha_critico = min(alpha_critico, bp['alpha_deg'])
                
        # Extraemos el valor del historial interno obligándolo a ser float puro
        T_final = float(solver.history_Tres[-1])

        # Post-procesamiento con criterios de diseño y diagnóstico funicular
        solver.post_process_MCFT(T_final, 30.0, 200000.0, 200.0, 200000.0, 400.0, alpha_critico)
        solver.diagnostico_funicular()
        solver.optimizar_dimensiones_circulares()

        # ============================================================
        # ESTUDIO DEL JACOBIANO SOBRE GEOMETRÍA OPTIMIZADA
        # ============================================================
        print("\n" + "="*70)
        print("ESTUDIO DEL JACOBIANO (SOBRE LA GEOMETRÍA OPTIMIZADA)")
        print("="*70)

        # Cálculo del Jacobiano numérico de G respecto a los ángulos de las ramas en la configuración final optimizada
        J, G0_vec, grad, norm_grad = compute_jacobian_alpha(solver, delta=1e-4)
        if J is not None and G0_vec is not None:
            n_ramas = J.shape[1]
            U, S, Vt = np.linalg.svd(J, full_matrices=True)
            rank = int(np.sum(S > 1e-6 * S[0]))   # tolerancia relativa
            nul_dim = 3 - rank

            print(f"  Gradiente J^T G = {grad}")
            print(f"  Norma del gradiente = {norm_grad:.6f} kN²/rad")
            print(f"Jacobiano J_G (3 x {n_ramas}):")
            print(f"  Rango = {rank}")
            print(f"  Dimensión del núcleo izquierdo Nul(J_G^T) = {nul_dim}")
            print(f"  Valores singulares: {S}")
            print(f"  Vector residuo G(alpha*) = {G0_vec} kN")
            print(f"  Norma total del residuo  : {np.linalg.norm(G0_vec):.6f} kN")

            # Interpretación:
            if nul_dim > 0:
                null_left = U[:, rank:]           # base de Nul(J^T)
                T_hiper_vec = null_left @ (null_left.T @ G0_vec)
                T_geo_vec   = G0_vec - T_hiper_vec
                print(f"  ||T_geo||   = {np.linalg.norm(T_geo_vec):.6f} kN")
                print(f"  ||T_hiper|| = {np.linalg.norm(T_hiper_vec):.6f} kN")
                print(f"  T_geo · T_hiper = {float(T_geo_vec @ T_hiper_vec):.2e} kN² (debe ≈ 0)")
            else:
                print("  Jacobiano rango completo: Nul(J^T) = {0}, T_hiper = 0.")

        # ============================================================
        # PRUEBA DE DESCENSO CON UN PASO DE GRADIENTE
        # ============================================================
        print("\n" + "="*70)
        print("PRUEBA DE DESCENSO CON UN PASO DE GRADIENTE (η variable)")
        print("="*70)

        # Probamos varios valores de η para verificar que el gradiente realmente apunta en dirección de descenso (reducción de T_res)
        best_eta, best_red = None, 0.0
        if J is not None and G0_vec is not None and grad is not None:
            branch_elems = [e for e in solver.elements if e.get('is_branch', False)]
            alphas_orig  = [e['branch_params']['alpha_rad'] for e in branch_elems]
            coords_orig  = [{'x': n['x'], 'y': n['y'], 'z': n['z']} for n in solver.nodes]
            T_old        = float(np.linalg.norm(G0_vec))

            print(f"Estado inicial (post-optimización): T_res = {T_old:.6f} kN")

            best_eta, best_T, best_red = None, float('inf'), 0.0
            for eta in [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
                delta_alpha = -eta * grad
                for i, e in enumerate(branch_elems):
                    bp      = e['branch_params']
                    a_new   = alphas_orig[i] + delta_alpha[i]
                    bp['alpha_rad'] = a_new
                    bp['alpha_deg'] = math.degrees(a_new)
                    solver.nodes[bp['node_inf']]['z'] = (
                        solver.nodes[bp['node_sup']]['z'] - bp['L_xy'] * math.tan(a_new)
                    )
                solver.ensamblar_sistema_global()
                solver.resolver()
                T_new = float(np.linalg.norm(solver.compute_T_res_vec()))
                red   = (T_old - T_new) / T_old * 100
                print(f"  η = {eta:.2e}: T_res = {T_new:.6f} kN, reducción = {red:.4f}%")
                if T_new < best_T:
                    best_T, best_eta, best_red = T_new, eta, red

                # Restaurar
                for idx, nd in enumerate(solver.nodes):
                    nd.update(coords_orig[idx])
                for i, e in enumerate(branch_elems):
                    e['branch_params']['alpha_rad'] = alphas_orig[i]
                    e['branch_params']['alpha_deg'] = math.degrees(alphas_orig[i])
        else:
            print("[Advertencia] No se puede ejecutar la prueba de descenso porque no se dispone del Jacobiano o del gradiente.")
            solver.ensamblar_sistema_global()
            solver.resolver()

        print("-"*70)
        if best_eta and best_red > 0:
            print(f"Mejor η = {best_eta:.2e} con reducción del {best_red:.4f}%")
            print("✓ Gradiente apunta en dirección de descenso.")
        else:
            print("✗ Ningún η redujo T_res. Revisar gradiente.")
        print("="*70)

        solver.generate_all_plots(resolution=20, margin_deg=10.0)
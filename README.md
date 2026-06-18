# Form-Finding Funicular de Estructuras Arbóreas 3D — n Ramas

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706928.svg)](https://doi.org/10.5281/zenodo.20706928)

Motor computacional para el form-finding funicular de estructuras arbóreas tridimensionales con número arbitrario de ramas, desarrollado como extensión del trabajo de referencia:

> Valencia Pérez, L. A. (2026). *Form-Finding Funicular de Estructuras Arbóreas Tridimensionales Mediante Relajación Angular Dinámica*. Zenodo. https://doi.org/10.5281/zenodo.20706928

Para el caso particular de 2 ramas, ver el repositorio complementario: [form-finding-arboreal-3D-2Ramas](https://github.com/LuisValencia4k/form-finding-arboreal-3D-2Ramas).

---

## Descripción

El repositorio contiene un único script monolítico (`Col_funic_n_ramas.py`) que implementa el **Principio de Funicularidad Ortogonal** para $n$ ramas arbitrarias. La clase principal es `FormFindingFEM`, acompañada de funciones auxiliares para construcción de geometría, experimentos paramétricos y cálculo del Jacobiano.

### Implementación

- **FEM de vigas Timoshenko** con 6 GDL por nodo, ensamble de rigidez global, matrices de transformación 3D y condiciones de borde configurables (empotramiento, articulación, capitel rígido)
- **Relajación Angular Dinámica (RAD)** con tasa de relajación exponencialmente decreciente, clipping geométrico para garantizar $\alpha_k \in (0°, 90°)$, y mecanismo de *early stopping* con *rollback* a la mejor configuración histórica
- **Jacobiano analítico** de $G(\boldsymbol{\alpha})$ por diferencias finitas, utilizado para la descomposición ortogonal $G = T_\text{geo} + T_\text{hiper}$ vía la Alternativa de Fredholm
- **Análisis espectral** del Jacobiano: valores singulares $\sigma_1 \geq \sigma_2 \geq \cdots$, número de condición $\kappa = \sigma_\text{max}/\sigma_\text{min}$ y rango estructural
- **Diagnóstico funicular** por energía de deformación: índice $I_f = U_\text{axial} / U_\text{total}$
- **Verificación MCFT** (Modified Compression Field Theory) de la fuerza de hendimiento en el nodo de bifurcación
- **Dimensionamiento morfológico** de fuste y ramas ahusadas por optimización de sección circular
- **Visualización** de morfología 3D, convergencia de $T_\text{res}$ y topografía del funcional $\Phi$ para todos los pares de ángulos

### Diferencias respecto al repo de 2 ramas

| Característica | 2 ramas | n ramas |
|---|---|---|
| Número de ramas | Fijo ($n=2$) | Arbitrario ($n \geq 2$) |
| Early stopping + rollback | No | Sí |
| Clipping geométrico | No | Sí |
| Historial de $G$, $V_\text{col}$, $\boldsymbol{\alpha}$ | No | Sí |
| Generación automática de topografías por pares | No | Sí (`generate_all_plots`) |
| Función de construcción general | No | `build_arboreal_structure_general` |
| Suite de experimentos paramétricos | No | `run_experiments` |

---

## Requisitos

```
Python >= 3.8
numpy
matplotlib
scipy
```

Instalación:

```bash
pip install numpy matplotlib scipy
```

---

## Uso

El script se ejecuta directamente. La geometría, cargas y parámetros se configuran mediante `build_arboreal_structure_general` en el bloque `if __name__ == '__main__'`:

```bash
python Col_funic_n_ramas.py
```

Para definir una estructura personalizada:

```python
branch_data_list = [
    {'Lxy': 1.5, 'phi_deg': 0.0,   'alpha0_deg': 35.0, 'b': 0.35, 'h': 0.35,
     'load': [0.0, 0.0, -600.0], 'top_node_z': 6.0},
    {'Lxy': 1.2, 'phi_deg': 120.0, 'alpha0_deg': 35.0, 'b': 0.30, 'h': 0.30,
     'load': [0.0, 0.0, -350.0], 'top_node_z': 5.0},
    {'Lxy': 1.0, 'phi_deg': 240.0, 'alpha0_deg': 35.0, 'b': 0.25, 'h': 0.25,
     'load': [0.0, 0.0, -200.0], 'top_node_z': 4.5},
]

solver = build_arboreal_structure_general(
    base_fix=[1,1,1,1,1,1],   # Base empotrada
    branch_data_list=branch_data_list,
    capitel_rigido=True
)

solver.form_finding_loop(max_iter=6000, tol=1e-2, w_max=0.3, w_min=0.01,
                         lam=0.1, patience=150)
solver.diagnostico_funicular()
solver.generate_all_plots()
```

### Parámetros del algoritmo RAD

| Parámetro | Descripción | Valor típico |
|---|---|---|
| `max_iter` | Iteraciones máximas | 6000 |
| `tol` | Tolerancia de convergencia en $T_\text{res}$ (kN) | 0.01 |
| `w_max`, `w_min` | Tasa de relajación inicial y final | 0.3, 0.01 |
| `lam` | Constante de amortiguamiento exponencial | 0.1 |
| `patience` | Iteraciones sin mejora antes del early stopping | 150 |

---

## Salidas

Al ejecutarse, el script produce:

- Convergencia de $T_\text{res}$ por iteración (escala semilogarítmica)
- Morfología arbórea 3D en el punto estacionario
- Descomposición $T_\text{geo}$ / $T_\text{hiper}$ con valores numéricos y porcentuales
- Espectro singular del Jacobiano, número de condición $\kappa$ y rango estructural
- Topografías de $\Phi(\alpha_i, \alpha_j)$ para todos los $\binom{n}{2}$ pares de ramas
- Cortante en columna padre por iteración
- Diagnóstico funicular por energía de deformación ($I_f$)
- Resumen de verificación MCFT

---

## Contexto teórico

El funcional minimizado es:

$$\Phi(\boldsymbol{\alpha}) = \tfrac{1}{2}\|G(\boldsymbol{\alpha})\|^2, \qquad G(\boldsymbol{\alpha}) = \sum_{k=1}^{n} \bigl(\mathbf{I} - \mathbf{u}_k \mathbf{u}_k^\top\bigr)\mathbf{F}_k$$

En el punto estacionario $\boldsymbol{\alpha}^*$, la componente $T_\text{geo}$ se anula. Para $n \geq 3$, el Jacobiano $J_G \in \mathbb{R}^{3 \times n}$ puede alcanzar rango 3, en cuyo caso $T_\text{hiper} = \mathbf{0}$ y la configuración es exactamente funicular. Para $n = 2$, el rango máximo es 2 y existe genéricamente residuo irreducible no trivial.

La degradación de convergencia de RAD para $n \geq 3$ se explica por el número de condición $\kappa \approx 66$–$96$, frente a $\kappa \approx 2.8$ para $n = 2$. El marco teórico completo — incluyendo la demostración formal de $G \in C^1(\mathcal{A})$, el Teorema de Condición Necesaria de Funicularidad y el análisis espectral — se desarrolla en el artículo complementario (en preparación).

---

## Licencia

MIT License. Ver archivo `LICENSE`.

---

## Autor

**Luis Alberto Valencia Pérez**  
Ingeniero Civil · Estructurista · Independiente · Oaxaca, México  
GitHub: [@LuisValencia4k](https://github.com/LuisValencia4k)

---

> *Este código es de naturaleza académica. Verificar resultados de forma independiente antes de cualquier aplicación estructural.*

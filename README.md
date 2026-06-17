# 🌳 ArborealFormFinding

**Motor de form‑finding funicular para estructuras arbóreas tridimensionales**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706928.svg)](https://doi.org/10.5281/zenodo.20706928)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/LuisValencia4k/form-finding-arboreal-3D)](https://github.com/LuisValencia4k/form-finding-arboreal-3D/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/LuisValencia4k/form-finding-arboreal-3D)](https://github.com/LuisValencia4k/form-finding-arboreal-3D/issues)

---

## Índice

- [Descripción](#-descripción)
- [Características principales](#-características-principales)
- [Marco teórico](#-marco-teórico)
- [Estructura del código](#-estructura-del-código)
- [Ejemplo de salida](#-ejemplo-de-salida)
- [Referencias](#-referencias)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)
- [Autor](#-autor)

---

## 📝 Descripción

**Motor de form‑finding funicular para estructuras arbóreas 3D con \(n\) ramas.** Optimiza la geometría vía FEM de vigas Timoshenko y Relajación Angular Dinámica (RAD). Incluye análisis espectral del Jacobiano, verificación estructural según MCFT y dimensionamiento automático de secciones de concreto.

Este repositorio contiene la implementación completa del algoritmo desarrollado en el artículo complementario:

> Valencia Pérez, L. A. (2026). *Marco Teórico para el Form‑Finding Funicular de Estructuras Arbóreas Tridimensionales*. DOI: [10.5281/zenodo.20706928](https://doi.org/10.5281/zenodo.20706928)

---

## Características principales

- **Modelado FEM 3D** con elementos viga de Timoshenko (axial, flexión, cortante y torsión).
- **Optimización morfológica** mediante Relajación Angular Dinámica (RAD) con clipping geométrico y *early stopping* con *rollback* al mejor histórico.
- **Precondicionamiento SVD** y **paso de Armijo** para acelerar la convergencia (variantes evaluadas en el artículo).
- **Análisis espectral** del Jacobiano: valores singulares, número de condición \(\kappa\) y diagnóstico de anisotropía.
- **Verificación estructural** según la Teoría de Campo de Compresión Modificada (MCFT) para puntales de concreto.
- **Diagnóstico funicular** basado en la energía de deformación axial vs. flexión/corte (índice funicular).
- **Dimensionamiento automático** de secciones circulares (fuste y ramas ahusadas) por pandeo, punzonamiento y flexocompresión.
- **Visualización 3D** de la geometría final y **mapas topográficos** del residuo en el espacio de ángulos (2D).

---

## Marco teórico

El algoritmo se basa en los siguientes fundamentos (desarrollados en el artículo asociado):

- **Operador de residuo** \(G(\boldsymbol{\alpha})\): suma de fuerzas perpendiculares a cada rama.
- **Funcional** \(\Phi = \frac{1}{2}\|G\|^2\) que se minimiza.
- **Descomposición ortogonal** \(G = T_{\text{geo}} + T_{\text{hiper}}\) (residuo geométrico + irreducible).
- **Teorema de condición necesaria**: si el Jacobiano tiene rango 3 en el punto estacionario, la configuración es funicular.
- **Análisis espectral**: el número de condición \(\kappa = \sigma_{\max}/\sigma_{\min}\) controla la velocidad de convergencia.

---

## Estructura del código


Col_funic_n_ramas.py
├── FormFindingFEM                # Clase principal
│   ├── add_node()                # Añade nodos con condiciones de borde y cargas
│   ├── add_element()             # Añade elementos viga (ramas, fuste, capitel)
│   ├── ensamblar_sistema_global()# Ensambla K_global y F_global
│   ├── resolver()                # Resuelve el sistema lineal
│   ├── compute_T_res_vec()       # Calcula el vector residuo G(alpha)
│   ├── form_finding_loop()       # Bucle principal de optimización RAD
│   ├── post_process_MCFT()       # Verificación según MCFT
│   ├── diagnostico_funicular()   # Índice y clasificación funicular
│   ├── optimizar_dimensiones_circulares() # Dimensionamiento de secciones
│   └── generate_all_plots()      # Genera gráficos de convergencia y morfología
│
├── build_arboreal_structure_general()  # Construye la geometría a partir de branch_data
├── compute_jacobian_alpha()            # Calcula el Jacobiano numérico J_G
└── run_experiments()                   # Experimentos de hiperestaticidad

## Ejemplo de salida
Iter 31: T_res = 4.5167e+00 kN | Φ = 1.0200e+01 kN² | V_fuste = 241.8720 kN
Mínimo irreducible final. Se agotaron los intentos de inercia.

--- RESULTADOS FINALES ---
Rama 2: α = 57.81°, Z_inf = 1.5000 m
Rama 4: α = 53.90°, Z_inf = 1.5000 m
Rama 6: α = 38.96°, Z_inf = 2.1130 m
Rama 8: α = 59.62°, Z_inf = 1.5000 m

=== POST-PROCESO: VERIFICACIÓN MCFT ===
Demanda neta de hendimiento (|T_res|) : 4.38 kN
Deformación transversal (ε_t)         : 0.000109 mm/mm
Factor de eficiencia (β_s)            : 1.000
Capacidad nominal del puntal (F_nn)   : 5100.00 kN
Capacidad vertical disponible         : 3206.52 kN
Cortante último actuante (V_u)        : 400.00 kN
Ratio Demanda/Capacidad               : 0.12
Estado de Verificación                : CUMPLE ✓

================================================
            DIAGNÓSTICO FUNICULAR               
================================================
T_res global remanente = 4.38 kN
Energía de deformación = 1.16 mJ (Bruto: 1.1600e-03 J)
------------------------------------------------
Energía axial          = 85.2 %
Energía flexión/corte  = 14.8 %

Índice funicular       = 0.852

Clasificación: [85.2%] CUASI FUNICULAR

## Referencias
Valencia Pérez, L. A. (2026). Form‑Finding Funicular de Estructuras Arbóreas Tridimensionales Mediante Relajación Angular Dinámica. DOI: 10.5281/zenodo.20706928

Barnes, M. R. (1999). Form finding and analysis of tension structures by dynamic relaxation. International Journal of Space Structures, 14(2), 89–104.

Collins, M. P. & Mitchell, D. (1991). Prestressed Concrete Structures. Prentice Hall.

Schek, H. J. (1974). The force density method for form finding and computation of general networks. Computer Methods in Applied Mechanics and Engineering, 3(1), 115–134.

Vecchio, F. J. & Collins, M. P. (1986). The modified compression field theory for reinforced concrete elements subjected to shear. ACI Structural Journal, 83(2), 219–231.

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerir mejoras, correcciones o nuevas funcionalidades.

## Licencia
Este proyecto se distribuye bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## Autor
Autor
Ing. Luis Alberto Valencia Pérez
Ingeniero Civil – Estructurista Independiente

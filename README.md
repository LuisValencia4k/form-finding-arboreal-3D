# ArborealFormFinding

**Motor de form‑finding funicular para estructuras arbóreas tridimensionales**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706928.svg)](https://doi.org/10.5281/zenodo.20706928)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Descripción

Este repositorio contiene la implementación completa del motor de **form‑finding funicular** para **estructuras arbóreas de concreto reforzado** con un número arbitrario de ramas (\(n \geq 2\)). El algoritmo optimiza la geometría de las ramas (ángulos de elevación) para minimizar el residuo global de fuerzas perpendiculares (\(T_{\text{res}}\)), logrando configuraciones **cuasi‑funiculares** con predominio de esfuerzos axiales y mínima flexión.

El código implementa el **Principio de Funicularidad Ortogonal** desarrollado en el artículo complementario:

> Valencia Pérez, L. A. (2026). *Marco Teórico para el Form‑Finding Funicular de Estructuras Arbóreas Tridimensionales*. DOI: [10.5281/zenodo.20706928](https://doi.org/10.5281/zenodo.20706928)

---

## Características principales

- **Modelado FEM 3D** con elementos viga de Timoshenko (rigidez axial, flexión, cortante y torsión).
- **Optimización morfológica** mediante Relajación Angular Dinámica (RAD) con clipping geométrico y *early stopping* con *rollback* al mejor histórico.
- **Precondicionamiento SVD** y **paso de Armijo** para acelerar la convergencia (variantes evaluadas en el artículo).
- **Análisis espectral** del Jacobiano: cálculo de valores singulares, número de condición \(\kappa\) y diagnóstico de anisotropía.
- **Verificación estructural** según la **Teoría de Campo de Compresión Modificada (MCFT)** para puntales de concreto.
- **Diagnóstico funicular** basado en la energía de deformación axial vs. flexión/corte (índice funicular).
- **Dimensionamiento automático** de secciones circulares (fuste y ramas ahusadas) por pandeo, punzonamiento y flexocompresión.
- **Visualización 3D** de la geometría final y **mapas topográficos** del residuo en el espacio de ángulos (2D).

---


---

## Requisitos e instalación

### Dependencias
- Python 3.8 o superior
- NumPy
- Matplotlib


### Ejemplo de salida
Iter 31: T_res = 4.5167e+00 kN | Φ = 1.0200e+01 kN² | V_fuste = 241.8720 kN
Mínimo irreducible final. Se agotaron los intentos de inercia.

--- RESULTADOS FINALES ---
Rama 2: α = 57.81°, Z_inf = 1.5000 m
Rama 4: α = 53.90°, Z_inf = 1.5000 m
Rama 6: α = 38.96°, Z_inf = 2.1130 m
Rama 8: α = 59.62°, Z_inf = 1.5000 m

=== POST-PROCESO: VERIFICACIÓN MCFT ===
Demanda neta de hendimiento (|T_res|) : 4.38 kN
...
Índice funicular = 0.852
Clasificación: [85.2%] CUASI FUNICULAR

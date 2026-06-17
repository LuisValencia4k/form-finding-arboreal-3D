[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706928.svg)](https://doi.org/10.5281/zenodo.20706928)
# Form-Finding Arboreal 3D

Motor de form-finding funicular para estructuras arbóreas 3D con n ramas. Optimiza geometría vía FEM de vigas Timoshenko y Relajación Angular Dinámica, con análisis espectral, verificación MCFT y dimensionamiento estructural.

## 🌳 Descripción General

Este proyecto implementa un motor avanzado de búsqueda de formas (form-finding) especializado en estructuras arbóreas tridimensionales. Utiliza principios de mecánica estructural funicular combinados con métodos numéricos sofisticados para encontrar geometrías óptimas en sistemas con múltiples ramas.

### Características Principales

- **Form-Finding Funicular**: Optimización de geometría basada en principios de cables y estructuras en tracción
- **FEM de Vigas Timoshenko**: Análisis mediante método de elementos finitos considerando cortante
- **Relajación Angular Dinámica**: Algoritmo iterativo para convergencia de geometría
- **Análisis Espectral**: Evaluación de modos de vibración y propiedades dinámicas
- **Verificación MCFT**: Validación mediante Teoría de Campo de Compresión Modificada
- **Dimensionamiento Estructural**: Cálculo automático de secciones transversales óptimas

## 📋 Requisitos

- Python 3.8+
- Dependencias (ver `requirements.txt`):
  - NumPy
  - SciPy
  - Matplotlib (para visualización)
  - [Otras dependencias según corresponda]

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/LuisValencia4k/form-finding-arboreal-3D.git
cd form-finding-arboreal-3D

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

```python
# Ejemplo básico de uso
from form_finding import ArborealStructure

# Crear estructura arbórea
structure = ArborealStructure(
    branches=5,
    height=10.0,
    loads=[...]  # Definir cargas
)

# Ejecutar form-finding
structure.optimize()

# Análisis adicionales
structure.spectral_analysis()
structure.verify_mcft()
structure.size_elements()

# Visualizar resultados
structure.plot()
```

## 🔧 Componentes Principales

### Form-Finding
- Algoritmo iterativo de búsqueda de formas equilibradas
- Soporte para múltiples ramas con conectividad arbitraria

### FEM Timoshenko
- Implementación de vigas Timoshenko (incluye deformación por cortante)
- Matriz de rigidez y vector de fuerzas

### Relajación Angular Dinámica
- Ajuste iterativo de ángulos nodales
- Convergencia controlada por tolerancia

### Análisis Espectral
- Cálculo de valores y vectores propios
- Análisis de frecuencias naturales

### Verificación MCFT
- Validación de resistencia mediante teoría de campo de compresión
- Determinación de capacidad a cortante y flexión

## 📊 Ejemplo de Salida

El programa genera:
- Geometría optimizada en formato 3D
- Diagramas de esfuerzos internos
- Análisis de modos de vibración
- Verificación de capacidades estructurales
- Dimensiones óptimas de elementos

## 📚 Documentación

Para información detallada sobre la teoría y métodos implementados, consulta la carpeta `docs/` (si disponible).

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**Luis Alberto Valencia Pérez**

- GitHub: [@LuisValencia4k](https://github.com/LuisValencia4k)

## 📞 Contacto y Soporte

Para preguntas o reportar issues, por favor abre una [issue](https://github.com/LuisValencia4k/form-finding-arboreal-3D/issues) en el repositorio.

---

**Nota**: Este es un proyecto de investigación especializado en optimización estructural y análisis numérico de sistemas arbóreos complejos.

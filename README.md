# Clases: Jupyter + VS Code

Este espacio es para desarrollar y explicar las clases localmente.

- `notebooks/`: apuntes, demostraciones y ejercicios interactivos (`.ipynb`).
- `src/`: funciones o módulos Python reutilizables (`.py`). `lora_chirp.py`
  contiene las funciones de modulación, demodulación, BER y ruido de la Clase 1.
- `src/experimentacion/`: scripts breves para probar los módulos sin alterar los
  notebooks.
- `src/experimentacion/graficos/`: imágenes generadas por los experimentos.
- `data/`: datos de entrada locales; no se versionan.
- `output/`: gráficos y resultados generados; no se versionan.

## Primer uso en VS Code

1. Abrí esta carpeta como espacio de trabajo en VS Code.
2. Seleccioná el intérprete de `.venv` como entorno Python.
3. Creá un notebook en `notebooks/` y elegí ese mismo *kernel*.
4. Guardá el código reutilizable en `src/` y usalo desde los notebooks.

Para las clases, Jupyter local dentro de VS Code es la opción recomendada: combina celdas ejecutables, gráficos, depuración y la asistencia de IA del editor en un único entorno.

## Reutilizar las funciones de la Clase 1

Desde un notebook abierto con la raíz del proyecto como carpeta de trabajo:

```python
from src.lora_chirp import modular_simbolo, demodular_simbolo, calc_error
```

También están disponibles `validar_simbolos`, `fase_sierra`, `agregar_ruido_awgn`
y `parametros_ruido_desde_snr`.

Para ejecutar la comprobación experimental de la cadena LoRa:

```bash
python src/experimentacion/validar_lora_chirp.py
```

"""Comprobación manual de la cadena LoRa de la Clase 1.

Ejecutar directamente con:
    python src/experimentacion/validar_lora_chirp.py
"""

from collections.abc import Sequence
from pathlib import Path
import sys

import matplotlib
import numpy as np
from numpy.typing import ArrayLike, NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Al ejecutar este archivo directamente, Python no incluye ``src`` en sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_chirp import (
    agregar_ruido_awgn,
    demodular_simbolo,
    modular_simbolo,
    parametros_ruido_desde_snr,
    validar_simbolos,
)


CARPETA_GRAFICOS = Path(__file__).with_name("graficos")
ANCHO_BANDA: float = 125_000
MuestrasReales = NDArray[np.float64]


def _guardar_vista_chirp(
    vistas: Sequence[tuple[MuestrasReales, MuestrasReales, MuestrasReales, MuestrasReales]],
    etiquetas: Sequence[str],
    limites_ms: MuestrasReales,
    nombre: str,
    descripcion: str,
    estilo: str,
) -> Path:
    """Guarda en una figura las vistas de todos los símbolos de una trama."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for (tiempo, real, imaginaria, frecuencia), etiqueta in zip(vistas, etiquetas):
        axes[0].plot(tiempo * 1e3, real, estilo, label=etiqueta)
        axes[1].plot(tiempo * 1e3, imaginaria, estilo)
        axes[2].plot(tiempo * 1e3, frecuencia / 1e3, estilo)

    axes[0].set(title=f"{descripcion}\nParte real", xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[1].set(title="Parte imaginaria", xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[2].set(
        title="Índice [(s + k) mod N] de la ecuación (3)",
        xlabel="Tiempo [ms]",
        ylabel="Índice × B/N [kHz]",
    )
    for ax in axes:
        for limite in limites_ms:
            ax.axvline(limite, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=min(len(etiquetas), 3), fontsize="small")
    ruta = CARPETA_GRAFICOS / nombre
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
    return ruta


def guardar_graficos(
    chirps: Sequence[ArrayLike],
    espectros: Sequence[ArrayLike],
    enviados: Sequence[int],
    estimados: Sequence[int],
    ancho_banda: float = ANCHO_BANDA,
) -> tuple[Path, Path]:
    """Guarda una trama de símbolos en gráficos temporales y espectrales comunes.

    Todos los símbolos aparecen uno después de otro en el mismo eje temporal.
    Las líneas verticales grises marcan dónde termina cada símbolo. Todas las
    curvas se obtienen de las ``N`` muestras de la ecuación (3) del artículo;
    no se introduce una extensión continua. El espectro superpone una curva
    por cada recepción para comparar los bins detectados.
    """
    CARPETA_GRAFICOS.mkdir(exist_ok=True)
    if not chirps:
        raise ValueError("debe haber al menos un chirp para graficar")
    if not (len(chirps) == len(espectros) == len(enviados) == len(estimados)):
        raise ValueError("chirps, espectros, enviados y estimados deben tener la misma longitud")

    chirps = [np.asarray(chirp) for chirp in chirps]
    espectros = [np.asarray(espectro) for espectro in espectros]
    N = len(chirps[0])
    if any(len(chirp) != N for chirp in chirps) or any(len(espectro) != N for espectro in espectros):
        raise ValueError("todos los chirps y espectros deben tener N muestras")

    k = np.arange(N)
    duracion_simbolo = N / ancho_banda
    limites_ms = np.arange(1, len(chirps)) * duracion_simbolo * 1e3
    etiquetas = [f"#{indice}: s={enviado}" for indice, enviado in enumerate(enviados, start=1)]

    # Cada vista se desplaza Ts para que toda la trama quede en un único gráfico.
    vistas_discretas = []
    for indice, (chirp, enviado) in enumerate(zip(chirps, enviados)):
        desplazamiento = indice * duracion_simbolo
        tiempo_discreto = desplazamiento + k / ancho_banda
        indice_ecuacion = np.mod(enviado + k, N) * ancho_banda / N
        vistas_discretas.append(
            (tiempo_discreto, chirp.real, chirp.imag, indice_ecuacion)
        )

    ruta_muestras = _guardar_vista_chirp(
        vistas_discretas,
        etiquetas,
        limites_ms,
        "chirp_muestras_discretas.png",
        f"Trama discreta: {len(chirps)} símbolos de N={N} muestras",
        "-",
    )

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for indice, (espectro, enviado, estimado) in enumerate(zip(espectros, enviados, estimados), start=1):
        energia = np.abs(espectro) ** 2
        linea, = ax.plot(
            np.arange(N), energia, label=f"#{indice}: enviado s={enviado}, FFT={estimado}"
        )
        ax.axvline(enviado, color=linea.get_color(), linestyle="--", alpha=0.6)
        ax.plot(estimado, energia[estimado], "x", color=linea.get_color(), markersize=8)
    ax.set(title="Espectros después del dechirp", xlabel="Bin de la FFT", ylabel="Energía")
    ax.set_xlim(-1, N)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ruta_espectro = CARPETA_GRAFICOS / "espectro_dechirp.png"
    fig.savefig(ruta_espectro, dpi=160)
    plt.close(fig)
    return ruta_muestras, ruta_espectro


def main() -> None:
    # Agregá o quitá palabras binarias aquí para simular una trama más corta o larga.
    simbolos = ["00100101", "11110000", "00000011"]
    sf = validar_simbolos(simbolos)
    N = 2**sf
    enviados = [int(bits, 2) for bits in simbolos]
    chirps = [modular_simbolo(enviado, np.arange(N), N) for enviado in enviados]

    potencia_senal = np.mean(np.abs(chirps[0]) ** 2)
    _, sigma = parametros_ruido_desde_snr(10, potencia_senal)
    generador = np.random.default_rng(2026)
    recibidas = [agregar_ruido_awgn(chirp, sigma, generador)[0] for chirp in chirps]
    resultados = [demodular_simbolo(recibida) for recibida in recibidas]
    estimados = [resultado[0] for resultado in resultados]
    espectros = [resultado[2] for resultado in resultados]

    assert estimados == enviados
    ruta_muestras, ruta_espectro = guardar_graficos(
        chirps, espectros, enviados, estimados
    )
    for bits, enviado, estimado in zip(simbolos, enviados, estimados):
        print(f"Enviado: {bits} (s={enviado}); detectado: s={estimado}")
    print(f"Gráficos guardados en: {ruta_muestras} y {ruta_espectro}")


if __name__ == "__main__":
    main()

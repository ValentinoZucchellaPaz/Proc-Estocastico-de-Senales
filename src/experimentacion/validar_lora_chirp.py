"""Comprobación manual de la cadena LoRa de la Clase 1.

Ejecutar directamente con:
    python src/experimentacion/validar_lora_chirp.py
"""

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


def _fase_sierra(
    tiempo: MuestrasReales, valor: int, N: int, ancho_banda: float
) -> MuestrasReales:
    """Devuelve la fase continua en ciclos para la curva teórica del chirp."""
    indice = valor + 2 * ancho_banda * np.asarray(tiempo)
    ciclos_completos = np.floor(indice / N)
    resto = np.mod(indice, N)
    area_de_sierra = (ciclos_completos * N**2 + resto**2) / 2
    return (area_de_sierra - valor**2 / 2) / (2 * N)


def _vista_teorica_chirp(
    valor: int, N: int, ancho_banda: float, muestras_por_muestra: int
) -> tuple[MuestrasReales, MuestrasReales, MuestrasReales, MuestrasReales]:
    """Construye una representación continua y sobreamuestreada para dibujar."""
    tiempo = np.linspace(0, N / ancho_banda, N * muestras_por_muestra + 1)
    fase = 2 * np.pi * _fase_sierra(tiempo, valor, N, ancho_banda)
    frecuencia = np.mod(valor + 2 * ancho_banda * tiempo, N) * ancho_banda / N
    return tiempo, np.cos(fase) / np.sqrt(N), np.sin(fase) / np.sqrt(N), frecuencia


def _guardar_vista_chirp(
    tiempo: MuestrasReales,
    real: MuestrasReales,
    imaginaria: MuestrasReales,
    frecuencia: MuestrasReales,
    nombre: str,
    descripcion: str,
    estilo: str,
) -> Path:
    """Guarda una sola representación del chirp para no mezclar conceptos."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(tiempo * 1e3, real, estilo)
    axes[0].set(title=f"{descripcion}\nParte real", xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[1].plot(tiempo * 1e3, imaginaria, estilo)
    axes[1].set(title="Parte imaginaria", xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[2].plot(tiempo * 1e3, frecuencia / 1e3, estilo)
    axes[2].set(title="Frecuencia instantánea", xlabel="Tiempo [ms]", ylabel="Frecuencia [kHz]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    ruta = CARPETA_GRAFICOS / nombre
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
    return ruta


def guardar_graficos(
    chirp: ArrayLike,
    espectro: ArrayLike,
    enviado: int,
    estimado: int,
    ancho_banda: float = ANCHO_BANDA,
    muestras_por_muestra: int = 16,
) -> tuple[Path, Path, Path]:
    """Guarda por separado la señal discreta y su modelo teórico suavizado.

    ``chirp_muestras_discretas.png`` contiene sólo las ``N`` muestras que salen
    del modulador y que efectivamente se transmiten. ``chirp_teorico.png`` no
    agrega muestras a la simulación: muestra el modelo continuo entre ellas
    para hacer visible la rampa de frecuencia. Se guardan en archivos distintos
    para no confundir la curva explicativa con una señal real de mayor resolución.

    ``muestras_por_muestra`` controla sólo la densidad de la línea teórica.
    Por ejemplo, 16 produce ``16 * N + 1`` puntos para verla suave.
    """
    CARPETA_GRAFICOS.mkdir(exist_ok=True)
    chirp = np.asarray(chirp)
    N = len(chirp)
    k = np.arange(N)

    # La vista discreta usa la salida del modulador, no una reconstrucción.
    tiempo_discreto = k / ancho_banda
    frecuencia_discreta = np.mod(enviado + 2 * k, N) * ancho_banda / N
    ruta_muestras = _guardar_vista_chirp(
        tiempo_discreto,
        chirp.real,
        chirp.imag,
        frecuencia_discreta,
        "chirp_muestras_discretas.png",
        f"Chirp discreto: N={N} muestras reales",
        "o-",
    )

    # La vista teórica se mantiene aquí —junto al código de gráficos— porque
    # no forma parte de la cadena de transmisión ni de la decisión por FFT.
    tiempo, real, imaginaria, frecuencia = _vista_teorica_chirp(
        enviado, N, ancho_banda, muestras_por_muestra
    )
    ruta_teorico = _guardar_vista_chirp(
        tiempo,
        real,
        imaginaria,
        frecuencia,
        "chirp_teorico.png",
        f"Modelo teórico del chirp ({muestras_por_muestra}× más puntos)",
        "-",
    )

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.stem(np.arange(len(espectro)), np.abs(espectro) ** 2, basefmt=" ")
    ax.axvline(enviado, color="tab:green", linestyle="--", label=f"enviado: s={enviado}")
    ax.axvline(estimado, color="tab:red", linestyle=":", label=f"detectado: s={estimado}")
    ax.set(title="Espectro después del dechirp", xlabel="Bin de la FFT", ylabel="Energía")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ruta_espectro = CARPETA_GRAFICOS / "espectro_dechirp.png"
    fig.savefig(ruta_espectro, dpi=160)
    plt.close(fig)
    return ruta_muestras, ruta_teorico, ruta_espectro


def main() -> None:
    bits = "00100101"
    sf = validar_simbolos([bits])
    N = 2**sf
    enviado = int(bits, 2)
    chirp = modular_simbolo(enviado, np.arange(N), N)

    potencia_senal = np.mean(np.abs(chirp) ** 2)
    _, sigma = parametros_ruido_desde_snr(10, potencia_senal)
    recibida, _ = agregar_ruido_awgn(chirp, sigma, np.random.default_rng(2026))
    estimado, _, espectro = demodular_simbolo(recibida)

    assert estimado == enviado
    ruta_muestras, ruta_teorico, ruta_espectro = guardar_graficos(
        chirp, espectro, enviado, estimado
    )
    print(f"Enviado: {bits} (s={enviado}); detectado: s={estimado}")
    print(f"Gráficos guardados en: {ruta_muestras}, {ruta_teorico} y {ruta_espectro}")


if __name__ == "__main__":
    main()

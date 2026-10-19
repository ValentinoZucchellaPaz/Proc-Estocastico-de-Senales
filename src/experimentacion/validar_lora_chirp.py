"""Comprobación manual de la cadena LoRa de la Clase 1.

Ejecutar directamente con:
    python src/experimentacion/validar_lora_chirp.py
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np

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


def guardar_graficos(tiempo, real, imaginaria, frecuencia, espectro, enviado, estimado):
    """Guarda las vistas temporal y espectral del experimento."""
    CARPETA_GRAFICOS.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(tiempo * 1e3, real)
    axes[0].set(title="Parte real del chirp", xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[1].plot(tiempo * 1e3, imaginaria)
    axes[1].set(title="Parte imaginaria del chirp", xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[2].plot(tiempo * 1e3, frecuencia / 1e3)
    axes[2].set(title="Frecuencia instantánea", xlabel="Tiempo [ms]", ylabel="Frecuencia [kHz]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    ruta_chirp = CARPETA_GRAFICOS / "chirp.png"
    fig.savefig(ruta_chirp, dpi=160)
    plt.close(fig)

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
    return ruta_chirp, ruta_espectro


def main():
    bits = "00100101"
    sf = validar_simbolos([bits])
    N = 2**sf
    enviado = int(bits, 2)
    chirp, tiempo, real, imaginaria, frecuencia = modular_simbolo(enviado, np.arange(N), N)

    potencia_senal = np.mean(np.abs(chirp) ** 2)
    _, sigma = parametros_ruido_desde_snr(10, potencia_senal)
    recibida, _ = agregar_ruido_awgn(chirp, sigma, np.random.default_rng(2026))
    estimado, _, espectro = demodular_simbolo(recibida)

    assert estimado == enviado
    ruta_chirp, ruta_espectro = guardar_graficos(
        tiempo, real, imaginaria, frecuencia, espectro, enviado, estimado
    )
    print(f"Enviado: {bits} (s={enviado}); detectado: s={estimado}")
    print(f"Gráficos guardados en: {ruta_chirp} y {ruta_espectro}")


if __name__ == "__main__":
    main()

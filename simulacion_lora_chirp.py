"""Simula de uno a cinco simbolos LoRa con la ecuacion (2) del articulo.

Los simbolos son cadenas binarias para conservar los ceros a la izquierda.
Requiere: numpy y matplotlib. Ejecutar con: python3 simulacion_lora_chirp.py
"""

import matplotlib
matplotlib.use("Agg")  # genera el PNG tambien en equipos sin interfaz grafica
import matplotlib.pyplot as plt
import numpy as np


B = 125_000  # ancho de banda [Hz]
MAX_SIMBOLOS = 5


def validar_simbolos(simbolos):
    """Devuelve SF y falla temprano si la lista no representa una trama valida."""
    if not isinstance(simbolos, (list, tuple)):
        raise TypeError("simbolos debe ser una lista o tupla de cadenas binarias")
    if not 1 <= len(simbolos) <= MAX_SIMBOLOS:
        raise ValueError(f"se admiten entre 1 y {MAX_SIMBOLOS} simbolos")
    if not all(isinstance(bits, str) and set(bits) <= {"0", "1"} for bits in simbolos):
        raise ValueError("cada simbolo debe ser una cadena formada solo por '0' y '1'")

    sf = len(simbolos[0])
    if sf == 0 or any(len(bits) != sf for bits in simbolos):
        raise ValueError("todos los simbolos deben tener la misma longitud y al menos un bit")
    return sf


def simular_lora(simbolos):
    """Modula y detecta una lista de 1 a 5 simbolos binarios de igual longitud."""
    sf = validar_simbolos(simbolos)
    N = 2**sf
    fs = B  # una muestra cada T = 1/B, como en el articulo
    Ts = N / B
    k = np.arange(N) # {1, 2, ..., 255}
    downchirp = np.exp(-1j * 2 * np.pi * k**2 / N)

    valores_enviados = [int(bits, 2) for bits in simbolos]
    valores_estimados = []
    chirps = []

    for valor in valores_enviados:
        # Ecuacion (2): c[k] = 1/sqrt(N) exp(j 2pi ((s+k) mod N) k / N).
        chirp = np.exp(1j * 2 * np.pi * ((valor + k) % N) * k / N) / np.sqrt(N)
        chirps.append(chirp)

        # Ecuacion (16): dechirp y FFT. El bin del maximo es el simbolo recibido.
        espectro = np.abs(np.fft.fft(chirp * downchirp))
        valores_estimados.append(int(np.argmax(espectro)))

    assert valores_estimados == valores_enviados
    _graficar(chirps, simbolos, valores_enviados, valores_estimados, k, fs, Ts, N)
    return valores_estimados


def _graficar(chirps, simbolos, enviados, estimados, k, fs, Ts, N):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    for indice, (chirp, bits) in enumerate(zip(chirps, simbolos)):
        tiempo = indice * Ts + k / fs
        etiqueta = f"#{indice + 1}: {bits}"
        axes[0].plot(tiempo * 1e3, chirp.real, marker=".", label=etiqueta)

        # El articulo representa frecuencias en [0, B], no en [-B/2, B/2].
        frecuencia = np.mod(np.angle(chirp[1:] * np.conj(chirp[:-1])), 2 * np.pi)
        frecuencia *= fs / (2 * np.pi)
        axes[1].step(tiempo[1:] * 1e3, frecuencia / 1e3, where="mid", label=etiqueta)

    limites_ms = np.arange(len(chirps) + 1) * Ts * 1e3
    for limite in limites_ms[1:-1]:
        axes[0].axvline(limite, color="gray", linestyle="--", alpha=0.6)
        axes[1].axvline(limite, color="gray", linestyle="--", alpha=0.6)

    axes[0].set(title="Parte real de los chirps transmitidos",
                xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=min(len(chirps), 3), fontsize="small")

    axes[1].set(title="Frecuencia instantanea (convencion del articulo: [0, B])",
                xlabel="Tiempo [ms]", ylabel="Frecuencia [kHz]")
    axes[1].grid(True, alpha=0.3)

    posiciones = np.arange(1, len(chirps) + 1)
    axes[2].plot(posiciones, enviados, "o-", color="crimson", label="enviado")
    axes[2].plot(posiciones, estimados, "x--", color="navy", label="detectado por FFT")
    for posicion, bits, valor in zip(posiciones, simbolos, enviados):
        axes[2].annotate(bits, (posicion, valor), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize="small")
    axes[2].set(title="Cada pico de FFT recupera el valor del simbolo enviado",
                xlabel="Numero de simbolo", ylabel="Valor decimal del simbolo",
                xlim=(0.7, len(chirps) + 0.3), ylim=(-1, N))
    axes[2].set_xticks(posiciones)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    output = "simulacion_lora_chirp.png"
    fig.savefig(output, dpi=160)
    print(f"SF={len(simbolos[0])}, B={B / 1e3:.0f} kHz, N={N}, Ts={Ts * 1e3:.3f} ms")
    for indice, (bits, enviado, estimado) in enumerate(zip(simbolos, enviados, estimados), start=1):
        print(f"Simbolo {indice}: {bits} -> s={enviado}; detectado={estimado}")
    print(f"Grafica guardada en: {output}")


if __name__ == "__main__":
    simular_lora(["00100101"])
    simular_lora(["00100101", "11110000"])
    # simular_lora(["00100101", "11110000", "00000011"])

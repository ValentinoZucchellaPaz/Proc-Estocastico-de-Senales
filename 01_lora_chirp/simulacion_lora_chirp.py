"""Simula de uno a cinco simbolos LoRa con la ecuacion (3) del articulo.

Los simbolos son cadenas binarias para conservar los ceros a la izquierda.
Requiere: numpy y matplotlib. Ejecutar con: python3 simulacion_lora_chirp.py
"""

import matplotlib
matplotlib.use("Agg")  # genera el PNG tambien en equipos sin interfaz grafica
import matplotlib.pyplot as plt
import numpy as np


B = 125_000  # ancho de banda [Hz]
MAX_SIMBOLOS = 5
MUESTRAS_POR_MUESTRA = 16  # solo para dibujar suavemente la onda temporal


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


def fase_sierra(tiempo, valor, N):
    """Fase continua cuya derivada es la sierra de frecuencia del chirp."""
    # En un simbolo Ts=N/B, 2*B*Ts=2*N: la sierra recorre dos veces los N bins.
    indice = valor + 2 * B * tiempo
    ciclos_completos = np.floor(indice / N)
    resto = np.mod(indice, N)
    area_de_sierra = (ciclos_completos * N**2 + resto**2) / 2
    return (area_de_sierra - valor**2 / 2) / (2 * N)


def modular_simbolo(valor, k, N):
    """Genera las muestras LoRa y una vista temporal sobreamostrada de la sierra."""
    # Ecuacion (3) del articulo: forma discreta usada por el modulador y la FFT.
    fase_discreta = ((valor + k) % N) * k / N
    chirp = np.exp(1j * 2 * np.pi * fase_discreta) / np.sqrt(N)

    Ts = N / B
    tiempo_vista = np.linspace(0, Ts, N * MUESTRAS_POR_MUESTRA + 1)
    # La pendiente 2*B/N viene de Delta(fase)=(valor+2*k+1)/N entre muestras.
    frecuencia_vista = np.mod(valor + 2 * B * tiempo_vista, N) * B / N
    fase = 2 * np.pi * fase_sierra(tiempo_vista, valor, N)
    onda_vista_real = np.cos(fase) / np.sqrt(N)
    onda_vista_imaginaria = np.sin(fase) / np.sqrt(N)
    return chirp, tiempo_vista, onda_vista_real, onda_vista_imaginaria, frecuencia_vista


def simular_lora(simbolos):
    """Modula y detecta una lista de 1 a 5 simbolos binarios de igual longitud."""
    sf = validar_simbolos(simbolos)
    N = 2**sf
    Ts = N / B
    k = np.arange(N)
    downchirp = np.exp(-1j * 2 * np.pi * k**2 / N)

    valores_enviados = [int(bits, 2) for bits in simbolos]
    valores_estimados = []
    vistas = []

    for valor in valores_enviados:
        chirp, tiempo_vista, real, imaginaria, frecuencia_vista = modular_simbolo(valor, k, N)
        vistas.append((tiempo_vista, real, imaginaria, frecuencia_vista))

        # Ecuacion (16): dechirp y FFT. El bin del maximo es el simbolo recibido.
        espectro = np.abs(np.fft.fft(chirp * downchirp))
        valores_estimados.append(int(np.argmax(espectro)))

    assert valores_estimados == valores_enviados
    _graficar(vistas, simbolos, valores_enviados, valores_estimados, Ts, N)
    return valores_estimados


def _graficar(vistas, simbolos, enviados, estimados, Ts, N):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    for indice, ((tiempo, real, imaginaria, frecuencia), bits, enviado, estimado) in enumerate(
        zip(vistas, simbolos, enviados, estimados)
    ):
        tiempo = indice * Ts + tiempo
        etiqueta = f"#{indice + 1}: {bits} | s={enviado}, FFT={estimado}"
        axes[0].plot(tiempo * 1e3, real, label=etiqueta)
        axes[1].plot(tiempo * 1e3, imaginaria, label=etiqueta)
        axes[2].plot(tiempo * 1e3, frecuencia / 1e3)

    limites_ms = np.arange(len(vistas) + 1) * Ts * 1e3
    for limite in limites_ms[1:-1]:
        axes[0].axvline(limite, color="gray", linestyle="--", alpha=0.6)
        axes[1].axvline(limite, color="gray", linestyle="--", alpha=0.6)
        axes[2].axvline(limite, color="gray", linestyle="--", alpha=0.6)

    axes[0].set(title="Parte real del chirp: cos(fase)",
                xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=min(len(vistas), 3), fontsize="small")

    axes[1].set(title="Parte imaginaria del chirp: sin(fase)",
                xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[1].grid(True, alpha=0.3)

    axes[2].set(title="Frecuencia instantanea: sierra entre 0 y B",
                xlabel="Tiempo [ms]", ylabel="Frecuencia [kHz]")
    axes[2].set_ylim(-1, B / 1e3 + 1)
    axes[2].grid(True, alpha=0.3)

    output = "simulacion_lora_chirp.png"
    fig.savefig(output, dpi=160)
    print(f"SF={len(simbolos[0])}, B={B / 1e3:.0f} kHz, N={N}, Ts={Ts * 1e3:.3f} ms")
    for indice, (bits, enviado, estimado) in enumerate(zip(simbolos, enviados, estimados), start=1):
        print(f"Simbolo {indice}: {bits} -> s={enviado}; detectado={estimado}")
    print(f"Grafica guardada en: {output}")


if __name__ == "__main__":
    simular_lora(["00100101"])
    # simular_lora(["00100101", "11110000"])
    # simular_lora(["00100101", "11110000", "00000011"])

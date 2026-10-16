"""Un simbolo LoRa de 8 bits usando exactamente la ecuacion (2) del articulo.

Requiere: numpy y matplotlib. Ejecutar con: python3 simulacion_lora_chirp.py
"""

import matplotlib
matplotlib.use("Agg")  # genera el PNG tambien en equipos sin interfaz grafica
import matplotlib.pyplot as plt
import numpy as np


BITS_TRANSMITIDOS = "00100101"  # cambiar por cualquier cadena de 8 bits
B = 125_000                    # ancho de banda [Hz]


def main():
    sf = len(BITS_TRANSMITIDOS)
    assert sf == 8 and set(BITS_TRANSMITIDOS) <= {"0", "1"}
    N = 2**sf
    simbolo = int(BITS_TRANSMITIDOS, 2)
    fs = B           # una muestra por 1/B, como en el articulo
    n = np.arange(N)
    t = n / fs

    # Ecuacion (2): c[k] = 1/sqrt(N) exp(j 2pi ((s+k) mod N) k / N).
    x = np.exp(1j * 2 * np.pi * ((simbolo + n) % N) * n / N) / np.sqrt(N)

    # El articulo usa la banda [0, B]. angle() devuelve [-pi, pi], por eso
    # se la lleva a [0, 2pi) antes de convertirla a Hz.
    f_inst = np.mod(np.angle(x[1:] * np.conj(x[:-1])), 2 * np.pi) * fs / (2 * np.pi)

    # La ecuacion (16) usa este down-chirp y luego una FFT para estimar s.
    downchirp = np.exp(-1j * 2 * np.pi * n**2 / N)
    dechirped = x * downchirp
    spectrum = np.abs(np.fft.fft(dechirped))
    spectrum /= spectrum.max()  # solo para que el pico se vea en escala 0 a 1
    simbolo_estimado = int(np.argmax(spectrum))
    assert simbolo_estimado == simbolo

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(t * 1e3, x.real, marker=".")
    axes[0].set(title=f"Parte real enviada: {BITS_TRANSMITIDOS} (s = {simbolo})",
                xlabel="Tiempo [ms]", ylabel="Amplitud")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(t[1:] * 1e3, f_inst / 1e3, where="mid")
    axes[1].set(title="Frecuencia instantanea (convencion del articulo: [0, B])",
                xlabel="Tiempo [ms]", ylabel="Frecuencia [kHz]")
    axes[1].grid(True, alpha=0.3)

    axes[2].stem(np.arange(N), spectrum, basefmt=" ")
    axes[2].axvline(simbolo, color="crimson", linestyle="--", label=f"s enviado = {simbolo}")
    axes[2].set(title=f"Dechirp + FFT: pico en {simbolo_estimado} = {BITS_TRANSMITIDOS}",
                xlabel="Bin FFT / valor del simbolo", ylabel="Magnitud normalizada",
                xlim=(-1, N))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    output = "simulacion_lora_chirp.png"
    fig.savefig(output, dpi=160)
    print(f"SF={sf}, B={B / 1e3:.0f} kHz, N={N}, Ts={N / B * 1e3:.3f} ms")
    print(f"Transmitido: {BITS_TRANSMITIDOS} -> s={simbolo}")
    print(f"Estimado por dechirp + FFT: s={simbolo_estimado}")
    print(f"Grafica guardada en: {output}")


if __name__ == "__main__":
    main()

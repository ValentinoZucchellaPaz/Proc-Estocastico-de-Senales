"""Funciones reutilizables para las simulaciones de chirps LoRa.

Un símbolo LoRa se representa con ``SF`` bits. Por eso hay ``N = 2**SF``
símbolos posibles, identificados por los enteros entre 0 y ``N - 1``. Cada
entero se transmite como un chirp: una señal compleja cuya frecuencia aumenta
linealmente y vuelve a cero al alcanzar el ancho de banda.

Las funciones de este módulo no dibujan gráficos. Separarlas de los notebooks
permite reutilizar la cadena de transmisión, canal y recepción en las clases
siguientes.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validar_simbolos(simbolos: list[str] | tuple[str, ...], max_simbolos: int = 5) -> int:
    """Valida una trama binaria y devuelve su factor de ensanchamiento (SF).

    Parameters
    ----------
    simbolos : list[str] | tuple[str, ...]
        Palabras binarias a transmitir. Todas deben tener la misma longitud:
        esa longitud es el ``SF``.
    max_simbolos : int
        Límite didáctico de símbolos por trama para las simulaciones de clase.

    Returns
    -------
    int
        El SF, que determina ``N = 2**SF`` y la cantidad de bins de la FFT.
    """
    # Validar antes de modular evita obtener resultados poco claros por entradas
    # como bits de longitudes diferentes o caracteres que no son 0 ni 1.
    if not isinstance(simbolos, (list, tuple)):
        raise TypeError("simbolos debe ser una lista o tupla de cadenas binarias")
    if not 1 <= len(simbolos) <= max_simbolos:
        raise ValueError(f"se admiten entre 1 y {max_simbolos} simbolos")
    if not all(isinstance(bits, str) and set(bits) <= {"0", "1"} for bits in simbolos):
        raise ValueError("cada simbolo debe ser una cadena formada solo por '0' y '1'")

    # Una palabra de SF bits selecciona exactamente uno de los 2**SF chirps.
    sf = len(simbolos[0])
    if sf == 0 or any(len(bits) != sf for bits in simbolos):
        raise ValueError("todos los simbolos deben tener la misma longitud y al menos un bit")
    return sf


def modular_simbolo(
    valor: int,
    k: ArrayLike,
    N: int,
) -> NDArray[np.complex128]:
    """Genera las ``N`` muestras discretas de un chirp LoRa.

    Parameters
    ----------
    valor : int
        Símbolo a transmitir, entre 0 y ``N - 1``.
    k : array_like
        Índices de muestra discretos; normalmente ``np.arange(N)``.
    N : int
        Cantidad de muestras por símbolo, igual a ``2**SF``.

    Returns
    -------
    numpy.ndarray
        Las muestras complejas que se transmiten y que recibe el demodulador.

    Notes
    -----
    Esta función no crea vectores de tiempo ni curvas suavizadas. Eso pertenece
    a la visualización en ``src/experimentacion``: separar ambos casos evita
    confundir la señal real de ``N`` muestras con su modelo teórico continuo.
    """
    # Convertir k a array permite aplicar la ecuación a todas las muestras a la vez.
    k = np.asarray(k)

    # Ecuación discreta del chirp LoRa. Sumar ``valor`` desplaza la sierra y
    # codifica el símbolo. El módulo N implementa el reinicio de frecuencia.
    fase_discreta = ((valor + k) % N) * k / N

    # exp(j·fase) produce una señal compleja de módulo constante. Dividir por
    # sqrt(N) normaliza la energía del símbolo a 1, independientemente del SF.
    return np.exp(1j * 2 * np.pi * fase_discreta) / np.sqrt(N)


def demodular_simbolo(
    recibida: ArrayLike,
) -> tuple[int, NDArray[np.complex128], NDArray[np.complex128]]:
    """Detecta un símbolo mediante dechirp y FFT.

    Se multiplica la señal recibida por un downchirp (el conjugado del chirp
    de referencia). Si se recibió un chirp LoRa ideal, esa multiplicación
    cancela la rampa de frecuencia y deja un tono. La posición de ese tono en
    la FFT es el valor entero del símbolo transmitido.

    Returns
    -------
    tuple[int, numpy.ndarray, numpy.ndarray]
        Símbolo estimado, señal dechirpeada y su espectro complejo.
    """
    # Forzar tipo complejo asegura que la multiplicación por el downchirp no
    # pierda su componente imaginaria, aunque la entrada fuese real.
    recibida = np.asarray(recibida, dtype=complex)
    N = len(recibida)
    if N == 0:
        raise ValueError("la señal recibida no puede estar vacía")

    # El downchirp es el conjugado del chirp de referencia de valor cero.
    # Su fase negativa revierte la rampa de frecuencia de la señal recibida.
    k = np.arange(N)
    downchirp = np.exp(-1j * 2 * np.pi * k**2 / N)
    dechirpeada = recibida * downchirp

    # La FFT separa los tonos por bin. El bin de mayor energía es la decisión
    # de máxima energía y, en el canal ideal, coincide con el símbolo enviado.
    espectro = np.fft.fft(dechirpeada)
    return int(np.argmax(np.abs(espectro) ** 2)), dechirpeada, espectro


def calc_error(
    enviados: Sequence[str] | ArrayLike,
    recibidos: Sequence[str] | ArrayLike,
    bits_por_simbolo: int | None = None,
) -> float:
    """Devuelve el BER: proporción de bits transmitidos incorrectamente.

    Sin ``bits_por_simbolo``, compara directamente dos secuencias de bits,
    como ``"001"`` y ``"011"``. Si se indica ``bits_por_simbolo``, interpreta
    las entradas como vectores de símbolos enteros y compara sus bits en
    representación binaria natural. Así la misma función sirve para un ejemplo
    pequeño y para la simulación vectorizada de una curva BER.
    """
    if len(enviados) != len(recibidos):
        raise ValueError("las dos secuencias deben tener la misma longitud")
    if len(enviados) == 0:
        raise ValueError("la secuencia de bits no puede estar vacía")
    if bits_por_simbolo is None:
        # True se interpreta como 1 y False como 0, de modo que la suma cuenta
        # los bits diferentes y dividir por el total da la proporción.
        return sum(a != b for a, b in zip(enviados, recibidos)) / len(enviados)

    if bits_por_simbolo <= 0:
        raise ValueError("bits_por_simbolo debe ser positivo")
    enviados = np.asarray(enviados, dtype=int)
    recibidos = np.asarray(recibidos, dtype=int)
    if enviados.ndim != 1 or recibidos.ndim != 1:
        raise ValueError("los símbolos deben ser vectores de una dimensión")
    limite = 2**bits_por_simbolo
    if np.any((enviados < 0) | (enviados >= limite) | (recibidos < 0) | (recibidos >= limite)):
        raise ValueError("los símbolos no caben en bits_por_simbolo bits")

    diferencias = np.bitwise_xor(enviados, recibidos)
    mascaras = 1 << np.arange(bits_por_simbolo)
    errores = np.count_nonzero((diferencias[:, None] & mascaras) != 0)
    return errores / (len(enviados) * bits_por_simbolo)


def agregar_ruido_awgn(
    senal: ArrayLike,
    sigma: float,
    generador: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Agrega ruido gaussiano blanco complejo CN(0, sigma² I) a una señal.

    ``sigma²`` es la potencia total del ruido complejo. Sus componentes real e
    imaginaria reciben varianza ``sigma² / 2`` para que, al sumarlas, la
    potencia sea precisamente ``sigma²``. Un generador opcional permite usar
    una semilla y repetir el mismo experimento.
    """
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma debe ser un número finito no negativo")
    # Crear el generador sólo si no se recibió uno mantiene la función útil tanto
    # para simulaciones aleatorias como para ejemplos reproducibles.
    generador = np.random.default_rng() if generador is None else generador
    senal = np.asarray(senal, dtype=complex)

    # Se generan dos gaussianas independientes: una para I (real) y otra para
    # Q (imaginaria). El factor 1/sqrt(2) reparte la potencia entre ambas.
    ruido = sigma / np.sqrt(2) * (
        generador.standard_normal(senal.shape) + 1j * generador.standard_normal(senal.shape)
    )
    return senal + ruido, ruido


def parametros_ruido_desde_snr(snr_db: float, potencia_senal: float) -> tuple[float, float]:
    """Calcula la potencia de ruido y sigma para una SNR objetivo en dB.

    La definición lineal es ``SNR = potencia_senal / potencia_ruido``. Como la SNR suele expresarse en decibeles, primero se convierte con
    ``SNR_lineal = 10**(snr_db / 10)`` y luego se despeja la potencia de ruido.
    """
    if not np.isfinite(snr_db):
        raise ValueError("snr_db debe ser un número finito")
    if not np.isfinite(potencia_senal) or potencia_senal <= 0:
        raise ValueError("potencia_senal debe ser un número finito positivo")

    # Despeje de potencia_ruido = potencia_senal / SNR_lineal.
    potencia_ruido = potencia_senal / 10 ** (snr_db / 10)

    # ``agregar_ruido_awgn`` recibe sigma, mientras que la SNR se formula con potencia. Para ruido complejo, potencia_ruido = sigma².
    return potencia_ruido, np.sqrt(potencia_ruido)

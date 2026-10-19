"""Funciones reutilizables para las simulaciones de chirps LoRa.

Un símbolo LoRa se representa con ``SF`` bits. Por eso hay ``N = 2**SF``
símbolos posibles, identificados por los enteros entre 0 y ``N - 1``. Cada
entero se transmite como un chirp: una señal compleja cuya frecuencia aumenta
linealmente y vuelve a cero al alcanzar el ancho de banda.

Las funciones de este módulo no dibujan gráficos. Separarlas de los notebooks
permite reutilizar la cadena de transmisión, canal y recepción en las clases
siguientes.
"""

import numpy as np


# 125 kHz es un ancho de banda habitual de LoRa. Se expresa en Hz para que
# tiempo, frecuencia y fase queden en unidades coherentes.
ANCHO_BANDA_POR_DEFECTO = 125_000

# Este sobreamuestreo se usa sólo para que las curvas temporales se vean suaves.
# El modulador y el demodulador trabajan con N muestras, no con estas muestras extra.
MUESTRAS_POR_MUESTRA_POR_DEFECTO = 16


def validar_simbolos(simbolos, max_simbolos=5):
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


def fase_sierra(tiempo, valor, N, ancho_banda=ANCHO_BANDA_POR_DEFECTO):
    """Calcula la fase continua, en ciclos, de la sierra de un chirp.

    La frecuencia instantánea es una rampa que se reinicia cada ``N`` bins.
    Como la fase es la integral de la frecuencia, esta función integra cada
    tramo de la rampa y suma los ciclos completos anteriores. El resultado se
    multiplica por ``2π`` antes de usarse en ``cos`` o ``exp``.
    """
    # ``indice`` expresa la posición de la rampa en bins. El factor 2 hace que
    # la frecuencia recorra la sierra dos veces durante Ts = N / ancho_banda.
    indice = valor + 2 * ancho_banda * np.asarray(tiempo)

    # Separamos los ciclos ya terminados del tramo actual para integrar una
    # rampa que puede haber cruzado el punto de reinicio varias veces.
    ciclos_completos = np.floor(indice / N)
    resto = np.mod(indice, N)

    # El área de una rampa lineal es x²/2. ``area_de_sierra`` es, por tanto,
    # la fase acumulada antes de normalizarla por el número de bins.
    area_de_sierra = (ciclos_completos * N**2 + resto**2) / 2

    # Restar ``valor² / 2`` hace que la fase temporal sea cero en t = 0,
    # incluso si el chirp comienza en un bin distinto de cero.
    return (area_de_sierra - valor**2 / 2) / (2 * N)


def modular_simbolo(
    valor: int,
    k,
    N:int,
    ancho_banda=ANCHO_BANDA_POR_DEFECTO,
    muestras_por_muestra=MUESTRAS_POR_MUESTRA_POR_DEFECTO,
):
    """Genera un chirp LoRa discreto y vectores para su vista temporal.

    Parameters
    ----------
    valor : int
        Símbolo a transmitir, entre 0 y ``N - 1``.
    k : array_like
        Índices de muestra discretos; normalmente ``np.arange(N)``.
    N : int
        Cantidad de muestras por símbolo, igual a ``2**SF``.
    ancho_banda : float
        Ancho de banda en Hz. Fija la duración ``Ts = N / ancho_banda``.
    muestras_por_muestra : int
        Sobreamuestreo empleado únicamente en las curvas de visualización.

    Returns
    -------
    tuple
        ``chirp`` es la señal discreta que se transmite; los cuatro valores
        restantes son tiempo, partes real e imaginaria y frecuencia para graficar.
    """
    # Convertir k a array permite aplicar la ecuación a todas las muestras a la vez.
    k = np.asarray(k)

    # Ecuación discreta del chirp LoRa. Sumar ``valor`` desplaza la sierra y
    # codifica el símbolo. El módulo N implementa el reinicio de frecuencia.
    fase_discreta = ((valor + k) % N) * k / N

    # exp(j·fase) produce una señal compleja de módulo constante. Dividir por
    # sqrt(N) normaliza la energía del símbolo a 1, independientemente del SF.
    chirp = np.exp(1j * 2 * np.pi * fase_discreta) / np.sqrt(N)

    # Las siguientes cantidades no intervienen en la demodulación: son una
    # representación continua y sobreamuestreada para entender y dibujar el chirp.
    tiempo_vista = np.linspace(0, N / ancho_banda, N * muestras_por_muestra + 1)
    frecuencia_vista = np.mod(valor + 2 * ancho_banda * tiempo_vista, N) * ancho_banda / N

    # ``fase_sierra`` devuelve ciclos; al multiplicar por 2π se obtiene radianes.
    fase = 2 * np.pi * fase_sierra(tiempo_vista, valor, N, ancho_banda)
    real = np.cos(fase) / np.sqrt(N)
    imaginaria = np.sin(fase) / np.sqrt(N)
    return chirp, tiempo_vista, real, imaginaria, frecuencia_vista


def demodular_simbolo(recibida):
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


def calc_error(bits_enviados, bits_recibidos):
    """Devuelve el BER: proporción de bits transmitidos incorrectamente.

    El BER (Bit Error Rate) compara cada posición de ambas secuencias. No es
    válido comparar secuencias de distinto largo porque ya no hay una relación
    uno a uno entre los bits.
    """
    if len(bits_enviados) != len(bits_recibidos):
        raise ValueError("las dos secuencias deben tener la misma longitud")
    if not bits_enviados:
        raise ValueError("la secuencia de bits no puede estar vacía")
    # True se interpreta como 1 y False como 0, de modo que la suma cuenta los
    # bits diferentes. Dividir por el total convierte ese conteo en proporción.
    return sum(a != b for a, b in zip(bits_enviados, bits_recibidos)) / len(bits_enviados)


def agregar_ruido_awgn(senal, sigma, generador=None):
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
        generador.standard_normal(senal.size) + 1j * generador.standard_normal(senal.size)
    )
    return senal + ruido, ruido


def parametros_ruido_desde_snr(snr_db, potencia_senal):
    """Calcula la potencia de ruido y sigma para una SNR objetivo en dB.

    La definición lineal es ``SNR = potencia_senal / potencia_ruido``. Como
    la SNR suele expresarse en decibeles, primero se convierte con
    ``SNR_lineal = 10**(snr_db / 10)`` y luego se despeja la potencia de ruido.
    """
    if not np.isfinite(snr_db):
        raise ValueError("snr_db debe ser un número finito")
    if not np.isfinite(potencia_senal) or potencia_senal <= 0:
        raise ValueError("potencia_senal debe ser un número finito positivo")

    # Despeje de potencia_ruido = potencia_senal / SNR_lineal.
    potencia_ruido = potencia_senal / 10 ** (snr_db / 10)

    # ``agregar_ruido_awgn`` recibe sigma, mientras que la SNR se formula con
    # potencia. Para ruido complejo, potencia_ruido = sigma².
    return potencia_ruido, np.sqrt(potencia_ruido)

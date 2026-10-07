import numpy as np


def dixon_filter(window, q_crit):
    data = np.asarray(window, dtype=float)
    m = data.size
    keep = np.ones(m, dtype=bool)
    # con menos de 4 datos la razón r22 no está definida
    if m < 4:
        return data.copy(), keep

    order = np.argsort(data)
    s = data[order]

    # extremo superior: (x(m) - x(m-2)) / (x(m) - x(3))
    den_up = s[-1] - s[2]
    if den_up > 0 and (s[-1] - s[-3]) / den_up > q_crit:
        keep[order[-1]] = False

    # extremo inferior, simétrico al anterior
    den_low = s[-3] - s[0]
    if den_low > 0 and (s[2] - s[0]) / den_low > q_crit:
        keep[order[0]] = False

    # se conserva el orden temporal original de las muestras que sobreviven
    return data[keep], keep


def gaussian_kernel(size, sigma):
    if size % 2 == 0 or sigma <= 0:
        raise ValueError("size debe ser impar y sigma positivo")
    half = size // 2
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_filter(series, sigma, size):
    data = np.asarray(series, dtype=float)
    kernel = gaussian_kernel(size, sigma)
    half = size // 2
    # relleno con el valor de borde para no perder muestras en los extremos
    padded = np.pad(data, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def kalman_filter(series, process_var, meas_var):
    data = np.asarray(series, dtype=float)
    if data.size == 0:
        raise ValueError("la serie está vacía")
    if process_var < 0 or meas_var <= 0:
        raise ValueError("varianzas inválidas")

    # estado inicial: mediana de la serie, robusta a picos
    x_hat = float(np.median(data))
    p = meas_var
    estimates = np.empty(data.size)
    for k, z in enumerate(data):
        # predicción (nodo estático, el estado no cambia)
        p_pred = p + process_var
        # corrección
        gain = p_pred / (p_pred + meas_var)
        x_hat = x_hat + gain * (z - x_hat)
        p = (1.0 - gain) * p_pred
        estimates[k] = x_hat
    return estimates

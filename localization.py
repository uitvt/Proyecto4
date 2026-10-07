import numpy as np


def rssi_to_distance(rssi, cfg):
    exponent = (cfg.rssi_d0 - rssi) / (10.0 * cfg.path_loss_exp)
    return float(cfg.d0 * 10.0 ** exponent)


# propagación de errores sobre la ecuación de distancia
def distance_variance(distance, rssi_var, n_samples, cfg):
    if n_samples < 1:
        raise ValueError("n_samples debe ser >= 1")
    factor = (np.log(10.0) / (10.0 * cfg.path_loss_exp)) ** 2
    return float(factor * distance ** 2 * rssi_var / n_samples)


def inverse_variance_weights(dist_vars, eps):
    return 1.0 / (np.asarray(dist_vars, dtype=float) + eps)


def multilaterate(
    anchors_xy,
    distances,
    ref_index=0,
    weights=None,
):
    xy = np.asarray(anchors_xy, dtype=float)
    d = np.asarray(distances, dtype=float)
    n = xy.shape[0]
    if n < 3 or d.size != n:
        raise ValueError("se requieren >= 3 anclas y una distancia por ancla")
    if not 0 <= ref_index < n:
        raise ValueError("ref_index fuera de rango")

    # todas las anclas menos la de referencia
    idx = np.array([i for i in range(n) if i != ref_index])
    xr, yr = xy[ref_index]
    xi, yi = xy[idx, 0], xy[idx, 1]

    a = np.column_stack((2.0 * (xr - xi), 2.0 * (yr - yi)))
    b = d[idx] ** 2 - d[ref_index] ** 2 - xi ** 2 - yi ** 2 + xr ** 2 + yr ** 2

    if weights is None:
        w = np.ones(idx.size)
    else:
        w = np.asarray(weights, dtype=float)[idx]

    # (A^T W A)^-1 A^T W B resuelto como mínimos cuadrados con sqrt(W), más estable
    sw = np.sqrt(w)
    try:
        sol, *_ = np.linalg.lstsq(a * sw[:, None], b * sw, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("no se pudo resolver el sistema de multilateración") from exc
    return sol

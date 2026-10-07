import numpy as np

from channel import RFChannel
from filters import dixon_filter, gaussian_filter, kalman_filter


def _rmse(values, reference):
    return float(np.sqrt(np.mean((np.asarray(values, dtype=float) - reference) ** 2)))


def validate_dixon(cfg, rng, trials=2000):
    # canal sin atípicos propios: el único pico es el que se inyecta a mano
    channel = RFChannel(cfg.with_updates(outlier_rate=0.0), rng)
    distance = 10.0
    detected = 0
    false_pos = 0

    for _ in range(trials):
        # ventana limpia (solo shadowing): cualquier descarte cuenta como falso positivo
        window, _ = channel.sample_window(distance, cfg.sigma_x)
        _, keep = dixon_filter(window, cfg.dixon_q_crit)
        if not keep.all():
            false_pos += 1

        # ventana con un atípico inyectado en una posición al azar
        window, _ = channel.sample_window(distance, cfg.sigma_x)
        pos = int(rng.integers(0, window.size))
        window[pos] += rng.choice([-1.0, 1.0]) * cfg.outlier_mag
        _, keep = dixon_filter(window, cfg.dixon_q_crit)
        if not keep[pos]:
            detected += 1

    return {
        "detection_rate": detected / trials,
        "false_positive_rate": false_pos / trials,
    }


def validate_smoothing(cfg, rng, trials=2000):
    channel = RFChannel(cfg, rng)
    distance = 10.0
    mu = channel.mean_rssi(distance)

    in_g, out_g, in_k, out_k = [], [], [], []
    # estimadores finales de cada ventana, para la comparación justa contra el promedio
    est_mean, est_gauss, est_kalman = [], [], []
    for _ in range(trials):
        window, _ = channel.sample_window(distance, cfg.sigma_x)
        cleaned, _ = dixon_filter(window, cfg.dixon_q_crit)

        # etapa gaussiana: error de las muestras antes y después de suavizar
        smooth = gaussian_filter(cleaned, cfg.gauss_sigma, cfg.gauss_kernel_size)
        in_g.append(_rmse(cleaned, mu))
        out_g.append(_rmse(smooth, mu))

        # etapa de Kalman: se compara la serie suavizada contra la trayectoria estimada
        r = max(float(np.var(cleaned, ddof=1)), cfg.kalman_r_floor)
        traj = kalman_filter(smooth, cfg.kalman_q, r)
        in_k.append(_rmse(smooth, mu))
        out_k.append(_rmse(traj, mu))

        est_mean.append(float(np.mean(cleaned)))
        est_gauss.append(float(np.mean(smooth)))
        est_kalman.append(float(traj[-1]))

    gauss_in, gauss_out = float(np.mean(in_g)), float(np.mean(out_g))
    kal_in, kal_out = float(np.mean(in_k)), float(np.mean(out_k))

    # error de cada estimador final respecto del valor sin ruido, sobre todas las ventanas
    rmse_mean = _rmse(np.array(est_mean), mu)
    rmse_gauss = _rmse(np.array(est_gauss), mu)
    rmse_kalman = _rmse(np.array(est_kalman), mu)
    return {
        # métrica por muestra: favorece a cualquier suavizado, se deja como referencia
        "gauss_sample_rmse_in": gauss_in,
        "gauss_sample_rmse_out": gauss_out,
        "gauss_sample_reduction_pct": 100.0 * (1.0 - gauss_out / gauss_in),
        "kalman_sample_rmse_in": kal_in,
        "kalman_sample_rmse_out": kal_out,
        "kalman_sample_reduction_pct": 100.0 * (1.0 - kal_out / kal_in),
        # métrica justa: estimador final contra el promedio simple de la ventana depurada
        "mean_estimator_rmse": rmse_mean,
        "gauss_estimator_rmse": rmse_gauss,
        "kalman_estimator_rmse": rmse_kalman,
        "gauss_vs_mean_pct": 100.0 * (1.0 - rmse_gauss / rmse_mean),
        "kalman_vs_mean_pct": 100.0 * (1.0 - rmse_kalman / rmse_mean),
    }


# valor crítico de la razón r22 por simulación bajo ruido gaussiano puro (una cola)
def estimate_dixon_critical(
    n, alpha, rng, trials=200000
):
    if n < 4 or not 0.0 < alpha < 1.0:
        raise ValueError("n >= 4 y alpha entre 0 y 1")
    data = np.sort(rng.normal(size=(trials, n)), axis=1)
    q_up = (data[:, -1] - data[:, -3]) / (data[:, -1] - data[:, 2])
    return float(np.quantile(q_up, 1.0 - alpha))


# barrido de detección y falsos positivos variando sigma, magnitud del atípico y Q crítico
def sweep_dixon(
    cfg,
    rng,
    sigmas=(2.0, 4.0, 6.0),
    magnitudes=(15.0, 25.0, 30.0),
    q_crits=(0.35, 0.45, 0.55),
    trials=1000,
):
    rows = []
    for sigma in sigmas:
        for mag in magnitudes:
            for q in q_crits:
                sc = cfg.with_updates(sigma_x=float(sigma), outlier_mag=float(mag), dixon_q_crit=float(q))
                res = validate_dixon(sc, rng, trials=trials)
                rows.append({"sigma": sigma, "outlier_mag": mag, "q_crit": q, **res})
    return rows

import numpy as np


class RFChannel:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    # RSSI medio sin ruido según el modelo log-distancia
    def mean_rssi(self, distance):
        # se acota la distancia mínima para no evaluar log10(0)
        d = max(distance, 0.1)
        return self.cfg.rssi_d0 - 10.0 * self.cfg.path_loss_exp * np.log10(d / self.cfg.d0)

    # ventana de muestras con shadowing gaussiano y picos impulsivos
    def sample_window(self, distance, sigma):
        cfg = self.cfg
        mu = self.mean_rssi(distance)
        noise = self.rng.normal(0.0, sigma, cfg.window_size)
        is_outlier = self.rng.random(cfg.window_size) < cfg.outlier_rate
        signs = self.rng.choice([-1.0, 1.0], cfg.window_size)
        window = mu + noise + is_outlier * signs * cfg.outlier_mag
        if cfg.quantize:
            window = np.round(window)
        return window, is_outlier

    # lectura de todas las anclas para un nodo (módulo de ingesta simulada)
    def read_all(self, node, anchors):
        windows = []
        for anchor in anchors:
            sigma = self.cfg.sigma_x if anchor.shadowing_sigma is None else anchor.shadowing_sigma
            dist = anchor.distance_to(node.position())
            window, _ = self.sample_window(dist, sigma)
            windows.append(window)
        return windows

    # lectura sin ruido ni cuantización, sirve para la prueba de conectividad base
    def read_clean(self, node, anchors):
        windows = []
        for anchor in anchors:
            mu = self.mean_rssi(anchor.distance_to(node.position()))
            windows.append(np.full(self.cfg.window_size, mu))
        return windows

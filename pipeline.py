import numpy as np

from filters import dixon_filter, gaussian_filter, kalman_filter
from localization import (
    distance_variance,
    inverse_variance_weights,
    multilaterate,
    rssi_to_distance,
)
from models import LinkEstimate


class VariantSpec:
    def __init__(self, name, use_dixon, use_gaussian, use_kalman, weighted, min_var_ref=False):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "use_dixon", use_dixon)
        object.__setattr__(self, "use_gaussian", use_gaussian)
        object.__setattr__(self, "use_kalman", use_kalman)
        object.__setattr__(self, "weighted", weighted)
        object.__setattr__(self, "min_var_ref", min_var_ref)

    def __setattr__(self, name, value):
        raise AttributeError("VariantSpec es inmutable")


# variantes del estudio de ablación: se agrega un bloque a la vez
# V3b separa el efecto de la ancla de referencia del efecto de los pesos
VARIANTS = (
    VariantSpec("V0", False, False, False, False),
    VariantSpec("V1", True, False, False, False),
    VariantSpec("V2", True, True, False, False),
    VariantSpec("V3", True, True, True, False),
    VariantSpec("V3b", True, True, True, False, min_var_ref=True),
    VariantSpec("V4", True, True, True, True, min_var_ref=True),
)


class LocalizationPipeline:
    def __init__(self, cfg, spec):
        self.cfg = cfg
        self.spec = spec

    def process_window(self, anchor_id, window):
        cfg, spec = self.cfg, self.spec
        data = np.asarray(window, dtype=float)
        if data.size < 4:
            raise ValueError("ventana demasiado corta")

        if spec.use_dixon:
            data, _ = dixon_filter(data, cfg.dixon_q_crit)

        # la varianza se calcula después de Dixon para que un pico aislado no penalice al ancla
        rssi_var = float(np.var(data, ddof=1)) if data.size > 1 else 0.0

        smooth = data
        if spec.use_gaussian:
            smooth = gaussian_filter(data, cfg.gauss_sigma, cfg.gauss_kernel_size)

        if spec.use_kalman:
            r = max(rssi_var, cfg.kalman_r_floor)
            rssi = float(kalman_filter(smooth, cfg.kalman_q, r)[-1])
        else:
            rssi = float(np.mean(smooth))

        dist = rssi_to_distance(rssi, cfg)
        dist_var = distance_variance(dist, rssi_var, data.size, cfg)
        return LinkEstimate(anchor_id, rssi, dist, dist_var, int(data.size))

    def estimate_links(self, anchors, windows):
        if len(anchors) != len(windows):
            raise ValueError("debe haber una ventana por ancla")
        return [self.process_window(a.anchor_id, w) for a, w in zip(anchors, windows)]

    def locate(self, anchors, windows):
        links = self.estimate_links(anchors, windows)
        anchors_xy = np.array([a.position() for a in anchors])
        distances = np.array([l.distance for l in links])

        dist_vars = np.array([l.distance_var for l in links])
        # referencia: ancla con menor incertidumbre, o la primera si la variante no la usa
        ref = int(np.argmin(dist_vars)) if self.spec.min_var_ref else 0

        if not self.spec.weighted:
            return multilaterate(anchors_xy, distances, ref_index=ref)

        weights = inverse_variance_weights(dist_vars, self.cfg.weight_eps)
        return multilaterate(anchors_xy, distances, ref_index=ref, weights=weights)

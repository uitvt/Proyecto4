import numpy as np


class Anchor:
    def __init__(self, anchor_id, x, y, shadowing_sigma=None):
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "shadowing_sigma", shadowing_sigma)

    def __setattr__(self, name, value):
        raise AttributeError("Anchor es inmutable")

    def position(self):
        return np.array([self.x, self.y], dtype=float)

    def distance_to(self, point):
        return float(np.hypot(self.x - point[0], self.y - point[1]))


class UnknownNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def position(self):
        return np.array([self.x, self.y], dtype=float)

    # distancia euclidiana entre la posición real y la estimada
    def error_to(self, estimate):
        return float(np.hypot(self.x - estimate[0], self.y - estimate[1]))


class LinkEstimate:
    def __init__(self, anchor_id, rssi, distance, distance_var, n_samples):
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "rssi", rssi)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "distance_var", distance_var)
        object.__setattr__(self, "n_samples", n_samples)

    def __setattr__(self, name, value):
        raise AttributeError("LinkEstimate es inmutable")


def build_anchors(cfg, sigma_overrides=None):
    overrides = sigma_overrides or {}
    anchors = []
    for idx, (x, y) in enumerate(cfg.anchors_xy):
        anchors.append(Anchor(idx, float(x), float(y), overrides.get(idx)))
    return anchors

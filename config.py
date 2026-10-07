class SimConfig:
    _defaults = {
        # lado del área cuadrada de simulación, en metros
        "area_size": 20.0,
        # 6 anclas: las cuatro esquinas y los puntos medios de los lados inferior y superior
        "anchors_xy": (
            (0.0, 0.0),
            (20.0, 0.0),
            (0.0, 20.0),
            (20.0, 20.0),
            (10.0, 0.0),
            (10.0, 20.0),
        ),
        # margen para que el nodo desconocido no caiga encima de una ancla
        "node_margin": 2.0,
        # parámetros del canal (modelo log-distancia)
        "d0": 1.0,
        "rssi_d0": -40.0,
        "path_loss_exp": 2.7,
        "sigma_x": 6.0,
        # RSSI cuantizado en dB enteros, como entregan los transceptores reales
        "quantize": True,
        # ventana de muestras y valores atípicos inyectados
        "window_size": 20,
        "outlier_rate": 0.05,
        "outlier_mag": 15.0,
        # parámetros de los filtros
        "dixon_q_crit": 0.450,
        "gauss_sigma": 1.0,
        "gauss_kernel_size": 5,
        "kalman_q": 0.01,
        "kalman_r_floor": 1.0,
        # regularización de los pesos (m^2)
        "weight_eps": 0.01,
        # corridas de Monte Carlo y semilla para reproducir
        "runs": 1000,
        "seed": 42,
    }

    def __init__(self, **overrides):
        unknown = overrides.keys() - self._defaults.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"argumentos inesperados: {names}")
        for name, default in self._defaults.items():
            object.__setattr__(self, name, overrides.get(name, default))
        self._validate()

    def __setattr__(self, name, value):
        raise AttributeError("SimConfig es inmutable")

    def with_updates(self, **overrides):
        values = {name: getattr(self, name) for name in self._defaults}
        values.update(overrides)
        return SimConfig(**values)

    def _validate(self):
        if len(self.anchors_xy) < 3:
            raise ValueError("se necesitan al menos 3 anclas")
        if self.window_size < 4:
            raise ValueError("la ventana debe tener al menos 4 muestras")
        if self.gauss_kernel_size % 2 == 0:
            raise ValueError("el núcleo gaussiano debe tener tamaño impar")
        if self.path_loss_exp <= 0 or self.d0 <= 0:
            raise ValueError("n y d0 deben ser positivos")
        if self.sigma_x < 0 or self.weight_eps <= 0 or self.runs < 1:
            raise ValueError("parámetros fuera de rango")
        if not 0.0 <= self.outlier_rate <= 1.0:
            raise ValueError("outlier_rate debe estar entre 0 y 1")

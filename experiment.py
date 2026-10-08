import numpy as np
from scipy.stats import wilcoxon

from channel import RFChannel
from models import UnknownNode, build_anchors
from pipeline import VARIANTS, LocalizationPipeline


def sample_node(cfg, rng):
    low = cfg.node_margin
    high = cfg.area_size - cfg.node_margin
    x, y = rng.uniform(low, high, 2)
    return UnknownNode(float(x), float(y))


def run_scenario(
    cfg,
    rng,
    specs=VARIANTS,
    sigma_overrides=None,
):
    anchors = build_anchors(cfg, sigma_overrides)
    channel = RFChannel(cfg, rng)
    pipelines = [LocalizationPipeline(cfg, s) for s in specs]
    errors = {s.name: np.full(cfg.runs, np.nan) for s in specs}

    for k in range(cfg.runs):
        node = sample_node(cfg, rng)
        # mismos datos para todas las variantes (diseño pareado)
        windows = channel.read_all(node, anchors)
        for pipe in pipelines:
            estimate = pipe.locate(anchors, windows)
            errors[pipe.spec.name][k] = node.error_to(estimate)
    return errors


# prueba de conectividad base: sin ruido el error debe ser prácticamente cero
def run_baseline_check(cfg, rng):
    anchors = build_anchors(cfg)
    channel = RFChannel(cfg, rng)
    pipe = LocalizationPipeline(cfg.with_updates(), VARIANTS[0])
    worst = 0.0
    for _ in range(50):
        node = sample_node(cfg, rng)
        windows = channel.read_clean(node, anchors)
        worst = max(worst, node.error_to(pipe.locate(anchors, windows)))
    return worst


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def summarize(errors):
    return {
        "rmse": rmse(errors),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
    }


def bootstrap_reduction_ci(
    base,
    test,
    rng,
    n_boot=2000,
):
    n = base.size
    reductions = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reductions[b] = 100.0 * (1.0 - rmse(test[idx]) / rmse(base[idx]))
    return float(np.percentile(reductions, 2.5)), float(np.percentile(reductions, 97.5))


def evaluate_scenario(label, param, errors, rng):
    base = errors["V0"]
    others = [name for name in errors if name != "V0"]
    rows = []
    for name, err in errors.items():
        row = {"scenario": label, "param": param, "variant": name}
        row.update(summarize(err))
        row["reduction_pct"] = 100.0 * (1.0 - row["rmse"] / rmse(base))
        row["ci_low"], row["ci_high"], row["p_adj"] = np.nan, np.nan, np.nan
        if name != "V0":
            row["ci_low"], row["ci_high"] = bootstrap_reduction_ci(base, err, rng)
            try:
                p = wilcoxon(base, err, alternative="greater").pvalue
            except ValueError:
                # todas las diferencias son cero, no hay prueba posible
                p = 1.0
            # corrección de Bonferroni por el número de comparaciones contra V0
            row["p_adj"] = float(min(1.0, p * len(others)))
        rows.append(row)
    return rows


# comparación pareada entre dos variantes cualesquiera (por ejemplo V3b contra V4)
def compare_pair(errors, base_name, test_name, rng, n_tests=1):
    base, test = errors[base_name], errors[test_name]
    low, high = bootstrap_reduction_ci(base, test, rng)
    try:
        p = float(wilcoxon(base, test, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    return {
        "base": base_name,
        "test": test_name,
        "reduction_pct": 100.0 * (1.0 - rmse(test) / rmse(base)),
        "ci_low": low,
        "ci_high": high,
        "p_value": p,
        # corrección de Bonferroni según el número de pares comparados
        "p_adj": min(1.0, p * n_tests),
    }


# S1: barrido de shadowing sin atípicos
def scenario_s1(cfg, rng, sigmas=(2, 4, 6, 8)):
    results = {}
    for sigma in sigmas:
        sc = cfg.with_updates(sigma_x=float(sigma), outlier_rate=0.0)
        results[f"sigma={sigma}"] = run_scenario(sc, rng)
    return results


# S2: sigma fijo en 6 dB y distinta proporción de atípicos
def scenario_s2(cfg, rng, rates=(0.0, 0.05, 0.10)):
    results = {}
    for rate in rates:
        sc = cfg.with_updates(sigma_x=6.0, outlier_rate=float(rate))
        results[f"outliers={int(rate * 100)}%"] = run_scenario(sc, rng)
    return results


# S3: cinco anclas con 4 dB y una degradada a 10 dB (la última de la lista)
def scenario_s3(cfg, rng):
    sc = cfg.with_updates(sigma_x=4.0, outlier_rate=0.0)
    degraded = len(cfg.anchors_xy) - 1
    return {"degraded_anchor": run_scenario(sc, rng, sigma_overrides={degraded: 10.0})}


# sensibilidad: se mueve un parámetro a la vez y se mide solo V4
def run_sensitivity(cfg, rng, grid):
    v4 = [s for s in VARIANTS if s.name == "V4"]
    rows = []
    for param, values in grid.items():
        for value in values:
            sc = cfg.with_updates(**{param: value})
            err = run_scenario(sc, rng, specs=v4)["V4"]
            rows.append({"param": param, "value": value, "rmse": rmse(err)})
    return rows

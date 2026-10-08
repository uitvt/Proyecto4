import argparse
import sys
from pathlib import Path

import numpy as np

from config import SimConfig
from experiment import (
    compare_pair,
    evaluate_scenario,
    run_baseline_check,
    run_sensitivity,
    sample_node,
    scenario_s1,
    scenario_s2,
    scenario_s3,
)
from reporting import plot_cdf, plot_layout, plot_rmse_vs_sigma, save_rows_csv
from validation import estimate_dixon_critical, sweep_dixon, validate_dixon, validate_smoothing


def print_rows(rows):
    print(f"{'scenario':<18}{'var':<5}{'RMSE':>8}{'median':>8}{'p90':>8}{'red.%':>8}{'p_adj':>9}")
    for r in rows:
        p = "-" if np.isnan(r["p_adj"]) else f"{r['p_adj']:.4f}"
        print(
            f"{r['param']:<18}{r['variant']:<5}{r['rmse']:>8.3f}{r['median']:>8.3f}"
            f"{r['p90']:>8.3f}{r['reduction_pct']:>8.1f}{p:>9}"
        )


def run_group(name, results, rng, out):
    rows = []
    for param, errors in results.items():
        rows += evaluate_scenario(name, param, errors, rng)
    print(f"\n== {name} ==")
    print_rows(rows)
    save_rows_csv(rows, out / f"{name}_metrics.csv")
    plot_rmse_vs_sigma(results, out / f"{name}_rmse.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Simulación de localización RSSI en WSN")
    parser.add_argument("--runs", type=int, default=1000, help="corridas por escenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--sensitivity", action="store_true", help="agrega el barrido de sensibilidad")
    args = parser.parse_args()

    try:
        cfg = SimConfig(runs=args.runs, seed=args.seed)
    except ValueError as exc:
        print(f"configuración inválida: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    rng = np.random.default_rng(cfg.seed)

    #  conectividad base sin perturbaciones
    worst = run_baseline_check(cfg, rng)
    print(f"Error máximo sin ruido (debe ser ~0): {worst:.2e} m")

    # Dixon se valida con sigma = 4 dB y atípicos de +-25 dB (OE1
    dix = validate_dixon(cfg.with_updates(sigma_x=4.0, outlier_mag=25.0), rng)
    smo = validate_smoothing(cfg, rng)
    print("\n== Validación de filtros ==")
    print(f"Dixon (sigma=4 dB, atípicos +-25 dB): detección={dix['detection_rate']:.3f}  falsos positivos={dix['false_positive_rate']:.3f}")
    print("Estimador final contra el promedio de la ventana depurada:")
    print(f"Promedio {smo['mean_estimator_rmse']:.3f} dB | Gaussiano {smo['gauss_estimator_rmse']:.3f} dB ({smo['gauss_vs_mean_pct']:.1f} %) | Kalman {smo['kalman_estimator_rmse']:.3f} dB ({smo['kalman_vs_mean_pct']:.1f} %)")
    save_rows_csv([{**dix, **smo}], out / "filter_validation.csv")

    # valor crítico de Dixon por simulación y barrido de detección
    q05 = estimate_dixon_critical(cfg.window_size, 0.05, rng)
    q10 = estimate_dixon_critical(cfg.window_size, 0.10, rng)
    print(f"\nQ crítico r22 simulado (n={cfg.window_size}, una cola): alpha=0.05 -> {q05:.3f}, alpha=0.10 -> {q10:.3f} (en config: {cfg.dixon_q_crit})")
    sweep = sweep_dixon(cfg, rng)
    save_rows_csv(sweep, out / "dixon_sweep.csv")
    print("\n== Barrido de Dixon (detección / falsos positivos) ==")
    print(f"{'sigma':<7}{'mag':<6}{'Qcrit':<7}{'detec':>8}{'falsos+':>9}")
    for r in sweep:
        print(f"{r['sigma']:<7}{r['outlier_mag']:<6}{r['q_crit']:<7}{r['detection_rate']:>8.3f}{r['false_positive_rate']:>9.3f}")

    # 3) escenarios S1, S2 y S3
    s1 = scenario_s1(cfg, rng)
    run_group("S1", s1, rng, out)
    s2 = scenario_s2(cfg, rng)
    run_group("S2", s2, rng, out)
    s3 = scenario_s3(cfg, rng)
    run_group("S3", s3, rng, out)

    # figuras de apoyo para el informe
    plot_cdf(s1["sigma=6"], "CDF del error, S1 (sigma = 6 dB)", out / "S1_cdf_sigma6.png")
    plot_cdf(s2["outliers=5%"], "CDF del error, S2 (5 % de atípicos)", out / "S2_cdf_outliers5.png")
    plot_cdf(s3["degraded_anchor"], "CDF del error, S3 (ancla degradada)", out / "S3_cdf_degraded.png")
    nodes = np.array([sample_node(cfg, rng).position() for _ in range(300)])
    plot_layout(np.array(cfg.anchors_xy, dtype=float), nodes, out / "layout.png")

    # 4) sensibilidad (opcional, tarda más)
    if args.sensitivity:
        grid = {
            "gauss_sigma": [0.5, 1.0, 2.0],
            "kalman_q": [0.001, 0.01, 0.1],
            "weight_eps": [0.001, 0.01, 0.1],
            "path_loss_exp": [2.2, 2.7, 3.2],
        }
        rows = run_sensitivity(cfg.with_updates(), rng, grid)
        save_rows_csv(rows, out / "sensitivity.csv")
        print("\n== Sensibilidad (RMSE de V4) ==")
        for r in rows:
            print(f"{r['param']:<14}{r['value']:<8}{r['rmse']:.3f}")

    # comparaciones pareadas para aislar el aporte de la referencia y de los pesos
    pair_rows = []
    for label, errs in (("S1 sigma=6", s1["sigma=6"]), ("S3 ancla degradada", s3["degraded_anchor"])):
        for base_name, test_name in (("V3", "V3b"), ("V3b", "V4"), ("V3", "V4")):
            pair_rows.append({"scenario": label, **compare_pair(errs, base_name, test_name, rng, n_tests=3)})
    save_rows_csv(pair_rows, out / "pairwise.csv")
    print("\n== Comparaciones pareadas ==")
    for r in pair_rows:
        print(f"{r['scenario']:<20}{r['base']:>4} -> {r['test']:<4}{r['reduction_pct']:>7.1f} %  IC95 [{r['ci_low']:.1f}, {r['ci_high']:.1f}]  p_adj={r['p_adj']:.4f}")
        
    # verificación de los objetivos específicos 1 y 2 del informe
    oe1 = dix["detection_rate"] >= 0.90 and dix["false_positive_rate"] <= 0.10
    s3_pair = [r for r in pair_rows if r["scenario"].startswith("S3") and r["base"] == "V3b" and r["test"] == "V4"][0]
    oe2 = s3_pair["reduction_pct"] >= 10.0 and s3_pair["p_adj"] < 0.05
    print(f"\nOE1 (Dixon: detección >= 90 % y falsos positivos <= 10 %): {'CUMPLE' if oe1 else 'NO CUMPLE'}")
    print(f"OE2 (V3b -> V4 con ancla degradada >= 10 %): {'CUMPLE' if oe2 else 'NO CUMPLE'}")

       # criterio de éxito del objetivo general
    # criterio de éxito del objetivo general
    ref = evaluate_scenario("S1", "sigma=6", s1["sigma=6"], rng)
    v4 = [r for r in ref if r["variant"] == "V4"][0]
    ok = v4["reduction_pct"] >= 15.0 and v4["p_adj"] < 0.05
    print(f"\nReducción V4 vs V0 con sigma=6 dB: {v4['reduction_pct']:.1f} % (p_adj={v4['p_adj']:.4f}) -> {'CUMPLE' if ok else 'NO CUMPLE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

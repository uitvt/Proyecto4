# Localización RSSI en WSN con ruido RF

Simulación Monte Carlo del flujo: Dixon -> filtro gaussiano -> Kalman escalar -> multilateración WLS.

## Instalación y ejecución

```bash
pip install -r requirements.txt
python main.py # 1000 corridas por escenario
python main.py --runs 200 # prueba rápida
python main.py --sensitivity # barrido de sensibilidad
```

Los resultados (CSV y figuras) se guardan en `results/`.

## Archivos principales

- `config.py`: parámetros del experimento (`SimConfig`)
- `models.py`: `Anchor`, `UnknownNode`, `LinkEstimate`
- `channel.py`: generación de ventanas RSSI con `RFChannel`
- `filters.py`: Dixon, filtro gaussiano y Kalman escalar
- `localization.py`: RSSI a distancia, varianza, pesos y multilateración
- `pipeline.py`: `LocalizationPipeline` y variantes V0 a V4
- `validation.py`: validación aislada de cada filtro
- `experiment.py`: escenarios S1, S2 y S3, métricas, bootstrap y Wilcoxon
- `reporting.py`: exportación de CSV y figuras
- `main.py`: punto de entrada

## Variantes del estudio (ablación)

- V0: promedio simple
- V1: + Dixon
- V2: + filtro gaussiano
- V3: + Kalman
- V3b: V3 usando como referencia el ancla de menor varianza, sin pesos
- V4: V3b + ponderación (algoritmo completo)

V3b permite separar el efecto de la referencia del efecto de los pesos.

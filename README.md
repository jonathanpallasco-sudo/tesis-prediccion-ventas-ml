<div align="center">

# Predicción de ventas con machine learning<br>en una distribuidora farmacéutica

**Modelos de predicción de ventas mediante técnicas de machine learning en una empresa retail dedicada a la distribución de productos farmacéuticos**

Jonathan Fernando Pallasco Catota · Tutor: Manuel Eugenio Morocho Cayamcela<br>
Maestría en Inteligencia de Negocios y Ciencia de Datos · Universidad de las Américas (UDLA) · Ecuador, 2026

![Python](https://img.shields.io/badge/Python-3.10.21-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-2_cuadernos-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)
![CTGAN](https://img.shields.io/badge/datos-sint%C3%A9ticos_CTGAN-6A1B9A)
![Licencia](https://img.shields.io/badge/licencia-MIT-lightgrey)

</div>

---

## Resumen de la tesis

En una empresa ecuatoriana de distribución farmacéutica, la meta mensual de venta de cada cliente en cada línea de producto se fija con el promedio de los últimos seis meses. Ese método no distingue una compra eventual de una recurrente ni considera el entorno económico. En el 57,4 % de los meses, el cliente no compra en una línea: la **demanda es intermitente**.

La tesis desarrolla y evalúa cinco algoritmos de machine learning (**Random Forest, perceptrón multicapa, XGBoost, LightGBM y CatBoost**) para predecir la venta neta mensual de cada relación cliente–línea con uno a seis meses de anticipación. Los compara con el método actual de la empresa y con **Croston–SBA**, el método clásico para demanda intermitente.

Como los datos son confidenciales, los 865.900 registros de facturación (enero de 2022 a septiembre de 2026) se **seudonimizaron con HMAC-SHA256**, y con ellos se generó una **base sintética** (cadena de Markov + CTGAN) que cumplió 18 de 20 criterios de fidelidad y privacidad. Todo el modelado se hizo sobre esa base sintética, junto con diez series macroeconómicas públicas, siguiendo la metodología **CRISP-DM**.

### Resultados

<div align="center">

| Método | WAPE en la prueba (abr–sep 2026) |
|:--|:--:|
| **Combinación Random Forest + Croston–SBA** | **0,643** |
| Croston–SBA | 0,656 |
| Random Forest | 0,669 |
| Método actual de la empresa | 0,727 |
| CatBoost | 0,742 |
| LightGBM | 0,752 |
| XGBoost | 0,769 |
| Perceptrón multicapa | 0,937 |

</div>

- **−8,4 puntos de error** frente al método actual con la combinación propuesta (IC 95 % bootstrap: −10,8 a −6,2).
- **78,7 %** de las ventas observadas cayó dentro del intervalo del 80 % del pronóstico.
- Lo que más pesa en el pronóstico son las **ventas recientes** y la **frecuencia de compra**; las variables macroeconómicas aportan menos de 2 puntos.
- **Limitación:** todos los métodos subestimaron el total del semestre. Los resultados provienen de la base sintética y deben validarse con datos reales antes de usarse en la empresa.

<div align="center">
<img src="Data/Analisis/outputs/figuras/V_wape_sesgo.png" width="820" alt="WAPE y sesgo del total pronosticado por método">
<br><sub>(a) Error por relación y mes; (b) sesgo del total pronosticado. Prueba de abril a septiembre de 2026.</sub>
</div>

---

## Arquitectura: dos entornos

```text
 ┌───────────────── ENTORNO SEGURO (solo en la empresa) ─────────────────┐
 │  data_real.xlsx  →  seudónimos HMAC-SHA256  →  panel cliente–línea–mes │
 │                  →  base sintética (Markov + CTGAN) y su validación    │
 └───────────────────────────────┬────────────────────────────────────────┘
                                 │  solo sale la base sintética
                                 ▼  y tablas agregadas
 ┌───────────────────────── ENTORNO DE ANÁLISIS ──────────────────────────┐
 │  base sintética + 10 series públicas  →  pronóstico directo h = 1…6     │
 │  →  5 algoritmos + Croston–SBA + método actual  →  evaluación,          │
 │      combinación con intervalo del 80 %, SHAP, LIME y permutación      │
 └────────────────────────────────────────────────────────────────────────┘
```

## Estructura del repositorio

```text
├── Entorno_seguro.ipynb             # seudonimización, panel, base sintética y validación
├── Entorno_análisis.ipynb           # análisis completo de los capítulos IV y V (14 pasos)
├── Data/
│   ├── Seguro/
│   │   ├── Macro/                   # series oficiales del INEC, el BCE y el MSP en su formato original
│   │   ├── exogenas_completadas.csv # petróleo WTI (EIA), índice ONI (NOAA) y meses recientes, con su fuente
│   │   └── outputs/publicable/      # base sintética y validación agregada (fidelidad y privacidad)
│   ├── Publico/                     # insumos del análisis: base_sintetica.csv y series alineadas
│   └── Analisis/outputs/
│       ├── tablas/                  # resultados R00–R31 (CSV y JSON)
│       └── figuras/                 # figuras de la tesis (PNG y SVG)
├── requirements.txt
└── LICENSE
```

## Cómo replicar

> [!IMPORTANT]
> Usa **Python 3.10.21**, la versión con la que se desarrolló y ejecutó el proyecto. Así CTGAN y las demás librerías se comportan igual que en la tesis.

```bash
# 1. Crear el entorno
conda create -n tesis_ctgan python=3.10.21 -y
conda activate tesis_ctgan

# 2. Instalar las librerías
pip install -r requirements.txt

# 3. Abrir los cuadernos desde la carpeta del repositorio
jupyter notebook
```

| Cuaderno | Quién puede ejecutarlo | Qué produce |
|:--|:--|:--|
| `Entorno_seguro.ipynb` | Solo la empresa: necesita `Data/Seguro/data_real.xlsx`, que no se publica | `Data/Publico/` y `Data/Seguro/outputs/publicable/` |
| `Entorno_análisis.ipynb` | Cualquier persona: usa los archivos de `Data/Publico/` incluidos aquí | `Data/Analisis/outputs/` (tablas y figuras de la tesis) |

Para reproducir los resultados, ejecuta **`Entorno_análisis.ipynb`** de principio a fin con los archivos de `Data/Publico/`. La semilla es `20240601`. No regeneres la base sintética: aun con la misma semilla, el entrenamiento de CTGAN cambia según la computadora y daría otra base.

### Verificación de la reproducción

El cuaderno de análisis se volvió a ejecutar completo en otra computadora (Linux, Python 3.10, versiones de `requirements.txt`) con los archivos de `Data/Publico/`. Tomó unos 47 minutos con 2 núcleos.

| Resultado | ¿Igual a la tesis? |
|:--|:--:|
| Análisis exploratorio, correlaciones, ACP y decisiones en validación (R01–R09) | ✅ idéntico |
| Random Forest, LightGBM, CatBoost, Croston–SBA y método de la empresa | ✅ idéntico |
| Combinación RF + Croston–SBA: WAPE 0,643, −8,4 pp [−10,8; −6,2], cobertura 78,7 % | ✅ idéntico |
| SHAP, LIME, serie total, escenario semestral y tipos de relación | ✅ idéntico |
| Réplica del estudio de Tía, diseño R1 | ✅ idéntico |
| Perceptrón multicapa (tesis 0,937 → 0,890) y XGBoost (0,769 → 0,759) | ⚠️ varía |
| Réplica de Tía, diseño R2: Random Forest (0,799 → 0,813) y red 15–10 (1,486 → 0,865) | ⚠️ varía |

Lo que varía depende de cálculos en paralelo y de rutinas numéricas que cambian levemente entre procesadores: las redes neuronales y XGBoost (que con otra computadora elige otra combinación de hiperparámetros) y los modelos con la configuración de Tía en el diseño R2. Por eso cambian sus filas en R10, R14–R16, R20, R23 (permutación de MLP y XGBoost) y R30. Las conclusiones se mantienen: Random Forest sigue siendo el mejor algoritmo y el perceptrón el peor, los boosting no se diferencian del método actual, la configuración de Tía queda por detrás de la propuesta y la combinación RF + Croston–SBA es la de menor error.

## Datos y privacidad

- **No se publican** la base real (`data_real.xlsx`), la clave HMAC, los archivos intermedios del entorno seguro ni la consulta de extracción. La celda de extracción se reemplazó por una descripción de sus columnas.
- Los modelos entrenados y las predicciones (`Data/Analisis/outputs/intermedios/`) tampoco se suben: se regeneran al ejecutar el cuaderno y superan el límite de tamaño de GitHub.
- La seudonimización no equivale a anonimización, y la generación con CTGAN tampoco garantiza privacidad por sí sola. En la validación, el 3,4 % de los registros sintéticos quedó más cerca de uno real que el percentil 5 de las distancias entre registros reales (umbral: 10 %), y ninguno es una copia.
- El tratamiento de los datos sigue la Ley Orgánica de Protección de Datos Personales del Ecuador (2021).

## Cita

```text
Pallasco Catota, J. F. (2026). Modelos de predicción de ventas mediante técnicas de machine learning
en una empresa retail dedicada a la distribución de productos farmacéuticos [Tesis de maestría,
Universidad de las Américas].
```

## Licencia

El código se publica bajo la licencia MIT (ver `LICENSE`). Las series de `Data/Seguro/Macro/` y `exogenas_completadas.csv` pertenecen a sus fuentes oficiales (INEC, Banco Central del Ecuador, Ministerio de Salud Pública, EIA y NOAA).

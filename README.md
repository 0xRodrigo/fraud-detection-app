# 💳 Detector de Fraudes Avanzado — Detalle Técnico y Justificación

## 📌 Propósito y Funcionamiento del Sistema

El **Detector de Fraudes Avanzado** es una aplicación interactiva que permite simular y analizar transacciones de tarjetas de crédito en tiempo real, demostrando cómo los algoritmos de *machine learning* pueden identificar posibles fraudes de manera automática y explicable. Su diseño está orientado tanto a usuarios técnicos como no técnicos, facilitando la exploración de los resultados y la interpretabilidad de las decisiones.

---

## 🎯 Hipótesis Experimental y Fundamento

- **Hipótesis Central:**  
  La detección automática de fraudes es viable utilizando únicamente atributos transformados (V1–V28) y el monto de la transacción (`Amount`), descartando la variable "Time", ya que representa valores relativos no aplicables de manera confiable en un entorno productivo o multi-institucional.

- **Justificación:**  
  En escenarios reales, la variable de tiempo es relativa y puede variar entre fuentes, lo que dificulta su uso y generalización. Al enfocarnos solo en variables derivadas de la transacción y el monto, el sistema replica condiciones de producción, priorizando la robustez y evitando sobreajuste a patrones espurios.

- **Origen de los Datos:**  
  Se emplea el conocido dataset [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), ampliamente utilizado en benchmarking de algoritmos de fraude.

---

## 🏗️ Pipeline y Modelos Empleados

### **1. Preprocesamiento**
- Solo se escala la variable `Amount` mediante `StandardScaler`.
- Las variables V1–V28 se mantienen en su forma original.
- Se excluye “Time” por ser un valor no generalizable.
- Los datos se reordenan antes de la predicción según el orden original de entrenamiento.

### **2. Modelos Aplicados**
- **Modelo Base:**  
  Random Forest entrenado sobre los datos originales (V1–V28 + Amount), sin rebalanceo artificial.
- **Modelo Optimizado:**  
  Random Forest ajustado con técnicas avanzadas de rebalanceo como **SMOTE** y variantes (BorderlineSMOTE, ADASYN, SMOTETomek, SMOTEENN, SMOTE_Aggressive). Además, incluye optimización de umbral para priorizar la detección de fraudes (recall).

### **3. Interpretabilidad**
- **SHAP:** Explicación global y local de la importancia de cada variable en la predicción.
- **LIME:** Interpretación local para instancias individuales, mostrando el aporte de cada atributo en cada transacción.

---

## 📊 Resultados Experimentales — Resumen

**Datos originales:**
- Total de registros: **284,807**
- Transacciones legítimas: **284,315** (99.83%)
- Fraudes: **492** (0.17%)

**Muestra usada para entrenamiento:**
- **50,000** registros (todos los fraudes + muestra de legítimas)
- **Entrenamiento:** 40,000  |  **Prueba:** 10,000

### **Resultados de los modelos principales**
| Modelo                         | AUC    | AP     | Recall (fraude) | Precision (fraude) |
|---------------------------------|--------|--------|-----------------|--------------------|
| LogisticRegression (original)   | 0.9810 | 0.8910 | 0.8571          | 0.8667             |
| RandomForest (original)         | 0.9715 | 0.9014 | 0.8571          | 0.8927             |
| RandomForest + SMOTE            | 0.9833 | 0.8944 | 0.8776          | 0.86               |

- **Mejor resultado balanceado**: RandomForest + SMOTE, Recall: **0.88**, Precision: **0.86**, F1-Score: **0.87**.
- **Estrategias experimentales:**  
  - Técnicas avanzadas de sampling.
  - Optimización de umbral de decisión para minimizar falsos negativos.
  - Ensembles conservadores y modelos *cost-sensitive*.

- **Mejor recall logrado (tolerando más falsos positivos):**  
  - **Recall:** 0.9286 (FN: 7, FP: 371) — con modelo *cost-sensitive*.

---

## 🛠️ Artefactos Generados para Producción

Todos los modelos y transformadores empleados han sido exportados para su reutilización y despliegue:

- `best_base_model.pkl` (modelo base)
- `amount_scaler.pkl` (escalador de monto)
- `best_strategy_model.pkl` (modelo optimizado)
- `strategy_scaler.pkl` (escalador para modelo optimizado)
- (opcional) `synth_stats.pkl` (para generación realista de ejemplos sintéticos)

**Importante:** Antes de predecir, los datos deben reordenarse para coincidir con el orden original de entrenamiento. El escalado se aplica solo a la columna `Amount`.

---

## 💡 Visualización y Experiencia de Usuario

- Permite experimentar con parámetros como el umbral de decisión y la probabilidad de fraude.
- Ofrece interpretabilidad visual para cada predicción, mostrando la contribución de las variables.
- Los resultados se acompañan de métricas clave (precision, recall, F1) y visualizaciones interactivas.
- La app puede generar ejemplos sintéticos realistas para pruebas en vivo sin exponer información sensible.

---

## 🔒 Consideraciones Éticas y de Seguridad

- Todos los datos empleados y generados son sintéticos o anonimizados.
- El sistema es **solo para demostración educativa**; no debe usarse para decisiones reales sin validación adicional.

---

## 🧑‍💻 Recursos y Más Información

- Dataset original: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Código fuente y detalles técnicos disponibles bajo consulta.
- Consultas y colaboración: contactar al autor del repositorio.

---

**¡Gracias por probar el Detector de Fraudes Avanzado!**  
Explora el funcionamiento, ajusta parámetros y observa el poder de la inteligencia artificial aplicada a la seguridad financiera.

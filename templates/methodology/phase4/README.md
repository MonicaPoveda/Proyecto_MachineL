# Fase 4 - Modelado (CRISP-DM)

## Objetivo

Entrenar y comparar diferentes algoritmos para clasificar URLs como legitimas (0) o maliciosas/phishing (1), utilizando las caracteristicas generadas en la fase 3.

Esta fase trabaja exclusivamente con `phase3_output/train.csv`. El archivo `phase3_output/test.csv` se reserva para la fase 5 (Evaluacion), para evitar utilizar los datos de prueba durante la seleccion y el ajuste de modelos.

## Modelos incluidos

1. **Regresion logistica:** modelo de referencia. Se aplica estandarizacion mediante un `Pipeline` porque el algoritmo es sensible a la escala de las caracteristicas.
2. **Random Forest:** combina varios arboles de decision y permite capturar relaciones no lineales.
3. **Gradient Boosting:** construye modelos secuenciales que intentan corregir los errores de los anteriores.

## Procedimiento

- Carga y valida el conjunto de entrenamiento.
- Comprueba que las caracteristicas sean numericas y no contengan valores nulos o infinitos.
- Utiliza validacion cruzada estratificada de 3 particiones.
- Busca hiperparametros con `GridSearchCV` para cada algoritmo.
- Selecciona la configuracion de cada algoritmo usando el F1 promedio de validacion cruzada.
- Registra precision, recall, ROC-AUC, exactitud (accuracy) y tasa de error como metricas complementarias. La tasa de error se calcula como 1 - accuracy.
- Guarda los modelos entrenados y los resultados para su revision.

La comparacion obtenida en esta fase es preliminar. La evaluacion definitiva sobre el conjunto de prueba y la decision final del modelo se documentan en la fase 5.

## Requisitos

Desde la raiz del proyecto, instala las dependencias:

```bash
python -m pip install -r requirements.txt
```

El proyecto ya incluye `pandas`, `numpy` y `scikit-learn` en `requirements.txt`. `joblib` se instala como dependencia de scikit-learn.

## Como ejecutar desde VS Code

1. Abre la carpeta raiz del repositorio.
2. Verifica que hayas ejecutado la fase 3 y que exista `phase3_output/train.csv`.
3. Abre una terminal en VS Code.
4. Ejecuta:

```bash
python templates/methodology/phase4/phase4_modeling.py
```

Opcionalmente, puedes indicar otras rutas o cambiar las particiones de validacion:

```bash
python templates/methodology/phase4/phase4_modeling.py --train phase3_output/train.csv --output phase4_output --cv-folds 3 --n-jobs -1
```

En Windows tambien puedes ejecutar el comando en PowerShell.

## Archivos generados

La ejecucion crea la carpeta `phase4_output/` en la raiz del proyecto:

- `model_comparison.csv`: resumen comparativo de los tres algoritmos, incluyendo `cv_accuracy_mean` y `cv_error_rate_mean`.
- `logistic_regression_search_results.csv`, `random_forest_search_results.csv` y `gradient_boosting_search_results.csv`: resultados de cada combinacion de hiperparametros.
- `modeling_metadata.json`: configuracion y datos generales del proceso.
- `models/`: modelos entrenados en formato Joblib.

Los archivos generados son resultados locales y no se agregan al repositorio.

## Criterio de comparacion

El F1 promedio de validacion cruzada se utiliza como metrica principal para ordenar los modelos, porque combina precision y recall. Tambien se reportan precision, recall y ROC-AUC para facilitar el analisis posterior.

No se declara aqui un modelo ganador definitivo: esa decision debe considerar la evaluacion sobre `test.csv` y los criterios del proyecto en la fase 5.


## Uso del modelo desde la plataforma

La ruta `/application/platform/` carga una sola vez el modelo con el mejor F1 promedio
de validacion cruzada y reutiliza ese modelo para las consultas siguientes. No vuelve
a entrenar al recibir una URL.

Para habilitar la prediccion, ejecuta la fase 4 y conserva localmente estos archivos:

- `phase4_output/model_comparison.csv`
- `phase4_output/modeling_metadata.json`
- `phase4_output/models/*.joblib`

Instala tambien las dependencias de `requirements.txt`, incluida `tldextract`,
necesaria para extraer las caracteristicas de las URLs.

La plataforma muestra la probabilidad estimada de la clase phishing cuando el modelo
la soporta y la tasa de error promedio de validacion cruzada. Esta tasa resume el
comportamiento del modelo durante la validacion; no representa el error de una URL
individual. La seleccion de fase 4 es preliminar y debe confirmarse con la evaluacion
del conjunto reservado `test.csv` en la fase 5.

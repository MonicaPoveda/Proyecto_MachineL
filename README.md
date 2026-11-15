# Phishing Risk Detection

Aplicación web en Flask que estima si una URL presenta características asociadas
con phishing. El clasificador usa características léxicas y estructurales de la
URL; no visita el sitio para producir la predicción ni garantiza que un enlace
sea seguro.

## Contenido

- [Descripción](#descripción)
- [Metodología y resultados](#metodología-y-resultados)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Reproducción local](#reproducción-local)
- [Uso de la aplicación](#uso-de-la-aplicación)
- [Despliegue en Render](#despliegue-en-render)
- [Datos y limitaciones](#datos-y-limitaciones)
- [Tecnologías](#tecnologías)

## Descripción

El proyecto implementa una aplicación de apoyo para analizar URLs. La
aplicación normaliza la entrada, extrae 31 características numéricas o
binarias y la clasifica como `Possible phishing` o `Likely legitimate`.
Cuando está disponible, también muestra una probabilidad estimada para la
clase phishing y una explicación basada en condiciones observadas en los
árboles del Random Forest.

La plataforma incluye registro e inicio de sesión. También ofrece una vista
previa de URLs HTTPS públicas directamente en el navegador, mediante un
`iframe` con sandbox. La vista previa no es parte de la predicción ni verifica
la seguridad del sitio.

## Metodología y resultados

El repositorio organiza sus páginas de metodología con las fases de CRISP-DM.
El informe del proyecto las presenta reorganizadas como seis fases de
CRISP-ML(Q): comprensión del negocio y los datos, ingeniería de datos,
ingeniería del modelo, evaluación, despliegue, y monitoreo y mantenimiento.

La preparación de datos combina indicadores IoC de tipo URL y FQDN como clase
positiva con ejemplos etiquetados como legítimos en `urls_leg.csv`. Después de
normalizar y deduplicar, el pipeline balancea las clases y genera una división
estratificada 80/20 con `random_state=42`.

Se comparan Regresión Logística, Random Forest y Gradient Boosting mediante
`GridSearchCV` y validación cruzada estratificada de tres pliegues. La selección
se realiza por F1 promedio. En el conjunto de prueba reservado, la evaluación
documentada para Random Forest reporta:

| Métrica | Resultado |
|---|---:|
| Exactitud | 95,95 % |
| Precisión de phishing | 97,35 % |
| Exhaustividad (recall) de phishing | 94,46 % |
| F1 | 95,88 % |
| ROC-AUC | 98,49 % |
| Tasa de falsos positivos | 2,57 % |

Estos resultados corresponden al conjunto de prueba balanceado del proyecto.
No demuestran por sí solos el desempeño en dominios nuevos, campañas futuras o
tráfico con una prevalencia distinta.

## Estructura del proyecto

```text
.
├── app.py
├── auth.py
├── db.py
├── requirements.txt
├── render.yaml
├── static/
│   └── datasets/
│       ├── indicators_of_compromise_ioc_20261106.csv
│       ├── urls_leg.csv
│       └── top-1m.csv
├── templates/
│   ├── application/
│   ├── auth/
│   └── methodology/
│       ├── phase3/
│       │   ├── features.py
│       │   └── phase3_preparation.py
│       └── phase4/
│           └── phase4_modeling.py
└── outputs/
    ├── phase3_output/
    │   ├── train.csv
    │   └── test.csv
    └── phase4_output/
        ├── model_comparison.csv
        ├── modeling_metadata.json
        └── models/
```

Los archivos de salida se regeneran durante la preparación y el entrenamiento.
El ranking `top-1m.csv` está en el repositorio, pero no se usa en el pipeline
actual de entrenamiento o inferencia.

## Requisitos

- Python 3.10 (Render está configurado con Python 3.10.12).
- `pip`.
- Git para clonar el repositorio.

Las dependencias están declaradas en `requirements.txt`. No todas tienen
versiones fijadas, por lo que las versiones instaladas pueden variar entre
entornos.

## Reproducción local

Los siguientes comandos se ejecutan desde la raíz del repositorio. En Windows,
PowerShell:

```powershell
git clone <URL_DEL_REPOSITORIO>
Set-Location Proyecto_MachineL

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python templates\methodology\phase3\phase3_preparation.py
python templates\methodology\phase4\phase4_modeling.py
python app.py
```

Si PowerShell impide activar el entorno virtual, se puede usar el intérprete
del entorno directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe templates\methodology\phase3\phase3_preparation.py
.\.venv\Scripts\python.exe templates\methodology\phase4\phase4_modeling.py
.\.venv\Scripts\python.exe app.py
```

En macOS o Linux, la preparación del entorno se realiza con:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_MachineL

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python templates/methodology/phase3/phase3_preparation.py
python templates/methodology/phase4/phase4_modeling.py
python app.py
```

La primera ejecución de la preparación de datos requiere que los archivos
IoC y `urls_leg.csv` estén en `static/datasets/`. El script de fase 3 crea
`outputs/phase3_output/`, incluidos `train.csv` y `test.csv`. El script de
fase 4 utiliza `train.csv` y guarda los modelos, la comparación y los
metadatos en `outputs/phase4_output/`.

Para iniciar Flask en modo de desarrollo, se puede usar también:

```bash
flask --app app run
```

La aplicación local se inicia en `http://127.0.0.1:5000`. El modo de desarrollo
no debe usarse como servidor de producción.

### Base de datos local

Si `DATABASE_URL` no está definida, `db.py` utiliza SQLite y crea `users.db`
en la raíz del proyecto. El archivo de base de datos local está excluido del
control de versiones mediante `.gitignore`.

## Uso de la aplicación

1. Abrir la aplicación local en el navegador.
2. Crear una cuenta desde la página de registro e iniciar sesión.
3. Ir a la plataforma de análisis e ingresar una URL.
4. Revisar la etiqueta, la probabilidad estimada y la explicación si están
   disponibles.
5. Si se usa la vista previa, recordar que se carga en el navegador del usuario
   y no determina si el sitio es seguro.

Un resultado `Likely legitimate` no es una certificación de seguridad. No se
deben ingresar contraseñas ni datos personales en sitios sospechosos o en la
vista previa.

## Despliegue en Render

El proyecto está desplegado en Render. El archivo `render.yaml` describe el
servicio web `machinegroup`, ejecutado con Gunicorn (`gunicorn app:app`), y la
base de datos PostgreSQL `machinegroup-db`. Render proporciona la cadena de
conexión mediante la variable `DATABASE_URL`; la configuración también genera
`SECRET_KEY` para la aplicación.

**Aplicación desplegada:** [https://cypherlink-1rnq.onrender.com](https://cypherlink-1rnq.onrender.com)

Para crear o actualizar un despliegue desde este repositorio:

1. Subir el repositorio a GitHub.
2. Crear o actualizar un servicio en Render conectado al repositorio y usar
   `render.yaml` como configuración.
3. Confirmar que el servicio web y la base de datos PostgreSQL se hayan creado
   y que las variables `DATABASE_URL` y `SECRET_KEY` estén disponibles para el
   servicio.
4. Verificar que los artefactos de `outputs/phase4_output/` estén incluidos en
   el despliegue. Si se vuelven a entrenar los modelos, conservar juntos el
   modelo, `model_comparison.csv` y `modeling_metadata.json`.
5. Probar registro, inicio de sesión y análisis de una URL después del
   despliegue.

No se deben guardar contraseñas, cadenas de conexión ni claves secretas en el
repositorio. El sistema de archivos predeterminado de Render puede ser
efímero; por ello, los artefactos del modelo deben estar incluidos en el
despliegue o guardados en almacenamiento persistente. La base PostgreSQL es
para las cuentas de usuario; no implementa por sí misma el monitoreo de
métricas del modelo.

## Datos y limitaciones

- El conjunto IoC se publica en el portal de datos abiertos de Colombia por
  MinTIC/ColCERT. La ficha del conjunto está disponible en
  [datos.gov.co](https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Indicadores-de-Compromiso-IoC/8zmi-ic7j/about_data).
- `urls_leg.csv` aporta los ejemplos etiquetados como legítimos. La
  interpretación de sus etiquetas depende de la procedencia y criterios usados
  para construir ese archivo.
- La clase positiva se construye con indicadores IoC de tipo URL y FQDN. Estos
  indicadores son etiquetas de amenaza de la fuente y no necesariamente casos
  de phishing confirmados de manera independiente.
- El modelo utiliza características léxicas y estructurales de la URL; no
  inspecciona el contenido de la página, redirecciones ni reputación en vivo.
- La partición de datos es aleatoria y estratificada, no separada por dominio
  o tiempo. Por ello, los resultados no garantizan generalización a dominios
  nuevos o campañas futuras.
- La clase «indeterminado» aparece como objetivo metodológico, pero la
  aplicación actual presenta una clasificación binaria.
- No se ha implementado monitoreo automático de *data drift* ni de métricas de
  producción.
- La vista previa acepta URLs HTTPS públicas y se carga en el navegador del
  usuario. Algunos sitios bloquean su inserción; una vista en blanco no indica
  que el sitio sea seguro o malicioso.

## Tecnologías

- Python, Flask y Gunicorn.
- pandas, NumPy y scikit-learn.
- Joblib para serializar los modelos.
- tldextract para analizar dominios.
- SQLite para desarrollo local y PostgreSQL en la configuración de Render.

## Referencias

Ministerio de Tecnologías de la Información y las Comunicaciones [MinTIC].
(2023). *Indicadores de Compromiso (IoC)* [Conjunto de datos]. Datos Abiertos
Colombia. [Ficha del conjunto de datos](https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Indicadores-de-Compromiso-IoC/8zmi-ic7j/about_data).

Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S.,
& Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning process model
with quality assurance methodology. *Machine Learning and Knowledge
Extraction, 3*(2), 392–413.
https://doi.org/10.3390/make3020020

Proyecto de Clustering — Segmentación de Clientes (Marketing)

Este proyecto implementa un estudio comparativo de tres algoritmos de aprendizaje no supervisado:

K-Means

Clustering Jerárquico

DBSCAN

Aplicados al caso real de segmentación de clientes en marketing, utilizando el dataset "Mall Customers".

El objetivo es identificar patrones de comportamiento entre clientes a partir de variables como edad, ingresos y nivel de gasto, para apoyar decisiones de marketing (por ejemplo: campañas dirigidas, diseño de promociones o segmentación estratégica).

Estructura del Proyecto
proyecto_clustering/
│── data/
│   └── mall_customers.csv
│
│── src/
│   ├── preprocessing.py
│   ├── kmeans_model.py
│   ├── hierarchical_model.py
│   ├── dbscan_model.py
│
│── results/
│   └── (Se guardarán automáticamente gráficas y métricas)
│
│── run_all.py
│── requirements.txt
│── README.md

Dataset Utilizado

Mall Customers Dataset
Columnas:

CustomerID (eliminada para clustering)

Gender

Age

Annual Income (k$)

Spending Score (1-100)

Se aplicó preprocesamiento:

Eliminación de ID

Codificación de género (Male = 0, Female = 1)

Estandarización con StandardScaler

Instalación

Clona el repositorio:

git clone https://github.com/tuusuario/proyecto_clustering.git
cd proyecto_clustering


Instala dependencias:

pip install -r requirements.txt

Ejecución del Proyecto

Corre todo con un solo comando:

python run_all.py


Esto generará:

Gráficos de K-Means, Jerárquico y DBSCAN

Dendrograma

Gráficos 2D/3D de clústeres

Métricas de evaluación:

Coeficiente de Silueta

Davies-Bouldin

Inertia (solo para K-Means)

Los resultados aparecerán en la carpeta /results/.

Algoritmos Implementados
1️. K-Means

Determinación óptima de K usando:

Método del codo

Silhouette score

Visualización 2D y 3D

2. Clustering Jerárquico

Método aglomerativo

Linkage: ward

Dendrograma incluido

Gráficos de clústeres

3. DBSCAN

Optimización visual de eps

Detección de ruido

Manejo de clústeres de densidad irregular

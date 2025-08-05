# ingest-data-proposal-ai

Aplicación para hacer la ingesta de datos para las propuestas de Red Hat, para su correcta ejecución se debe tener un llama-stack-server corriendo.

## Setup local

Inicializar el ambiente de Python

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

Instalar las dependencias

```bash
pip install -r requirements.txt
```

Ejecutar localmente

```bash
python ingest_data_rag.py
```

## Setup Notebook

Para usar dentro de un Notebook clone este repostorio y use ingest_data.ipynb
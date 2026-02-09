# Salary Prediction API - Trabajo Final

API REST con Flask para predecir salarios en Data Science usando Machine Learning.

## Descripción

Este proyecto implementa un modelo de Machine Learning para predecir salarios de profesionales en el área de Data Science basándose en diferentes características como experiencia, tipo de empleo, tamaño de empresa, etc.

## Estructura del Proyecto

```
trabajo-final/
├── app.py                    # API Flask principal
├── requirements.txt          # Dependencias del proyecto
├── .env                      # Variables de entorno
├── model/
│   ├── model_salaries.pkl    # Modelo entrenado
│   ├── scaler_X_salaries.pkl # Scaler para features
│   └── scaler_y_salaries.pkl # Scaler para target
└── templates/
    └── index.html            # Interfaz web
```

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno en `.env`:
```
DATABASE_URL=sqlite:///salaries.db
```

## Uso

### Iniciar el servidor
```bash
python app.py
```

El servidor iniciará en `http://localhost:5000`

### Interfaz Web
Accede a `http://localhost:5000` para usar la interfaz gráfica de predicción.

### Endpoints API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Interfaz web de predicción |
| GET | `/api` | Información de la API |
| POST | `/predict` | Predecir salario (sin guardar) |
| POST | `/salary` | Crear predicción y guardar en BD |
| GET | `/salary` | Obtener todas las predicciones |
| GET | `/salary/<id>` | Obtener predicción por ID |
| PUT | `/salary/<id>` | Actualizar predicción |
| DELETE | `/salary/<id>` | Eliminar predicción |

### Ejemplo de Predicción

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "work_year": 2024,
    "experience_level": 3,
    "employment_type": 2,
    "company_size": 1,
    "job_title": 114,
    "employee_residence": 91,
    "remote_ratio": 0,
    "company_location": 86
  }'
```

### Codificación de Variables

**experience_level:**
- 0: Entry Level
- 1: Junior
- 2: Mid-Level
- 3: Senior

**employment_type:**
- 0: Freelance
- 1: Part-Time
- 2: Full-Time
- 3: Contract

**company_size:**
- 0: Small (S)
- 1: Medium (M)
- 2: Large (L)

**remote_ratio:**
- 0: Presencial
- 50: Híbrido
- 100: Remoto

## Tecnologías Utilizadas

- Flask
- SQLAlchemy
- Marshmallow
- Scikit-learn
- Joblib
- NumPy

## Autor

Trabajo Final - Machine Learning

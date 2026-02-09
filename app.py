from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from decouple import config
from flask_cors import CORS

import joblib
import numpy as np

# Load the model and scalers
model = joblib.load('./model/model_salaries.pkl')
sc_x = joblib.load('./model/scaler_X_salaries.pkl')
sc_y = joblib.load('./model/scaler_y_salaries.pkl')

app = Flask(__name__)
CORS(app)

#### SQLALCHEMY CONFIGURATION ####
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Salary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    work_year = db.Column(db.Integer, nullable=False)
    experience_level = db.Column(db.Integer, nullable=False)
    employment_type = db.Column(db.Integer, nullable=False)
    job_title = db.Column(db.Integer, nullable=False)
    employee_residence = db.Column(db.Integer, nullable=False)
    remote_ratio = db.Column(db.Integer, nullable=False)
    company_location = db.Column(db.Integer, nullable=False)
    company_size = db.Column(db.Integer, nullable=False)
    predicted_salary = db.Column(db.Float, nullable=True)
    
    def __init__(self, work_year, experience_level, employment_type, job_title, 
                 employee_residence, remote_ratio, company_location, company_size):
        self.work_year = work_year
        self.experience_level = experience_level
        self.employment_type = employment_type
        self.job_title = job_title
        self.employee_residence = employee_residence
        self.remote_ratio = remote_ratio
        self.company_location = company_location
        self.company_size = company_size

### CREATE A SCHEMA TO SERIALIZE DATA
ma = Marshmallow(app)
class SalarySchema(ma.Schema):
    id = ma.Integer()
    work_year = ma.Integer()
    experience_level = ma.Integer()
    employment_type = ma.Integer()
    job_title = ma.Integer()
    employee_residence = ma.Integer()
    remote_ratio = ma.Integer()
    company_location = ma.Integer()
    company_size = ma.Integer()
    predicted_salary = ma.Float()

## REGISTER THE TABLE IN THE DATABASE
db.create_all()

def predict_salary(work_year, experience_level, employment_type, job_title, 
                   employee_residence, remote_ratio, company_location, company_size):
    features = np.array([[work_year, experience_level, employment_type, job_title, 
                          employee_residence, remote_ratio, company_location, company_size]])
    features_scaled = sc_x.transform(features)
    prediction_scaled = model.predict(features_scaled)
    prediction = sc_y.inverse_transform(prediction_scaled.reshape(1, -1))
    salary = round(float(prediction[0][0]) * 1000, 2)
    return salary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api')
def api_info():
    context = {
        'title': 'SALARY PREDICTION API V1.0',
        'message': 'Welcome to the Data Science Salary Prediction API',
        'endpoints': {
            'POST /salary': 'Crear nueva prediccion de salario',
            'GET /salary': 'Obtener todas las predicciones',
            'GET /salary/<id>': 'Obtener prediccion por ID',
            'PUT /salary/<id>': 'Actualizar prediccion',
            'DELETE /salary/<id>': 'Eliminar prediccion'
        }
    }
    return jsonify(context)

@app.route('/salary', methods=['POST'])
def set_data():
    work_year = request.json['work_year']
    experience_level = request.json['experience_level']
    employment_type = request.json['employment_type']
    job_title = request.json.get('job_title', 114)
    employee_residence = request.json.get('employee_residence', 91)
    remote_ratio = request.json.get('remote_ratio', 0)
    company_location = request.json.get('company_location', 86)
    company_size = request.json['company_size']
    
    predicted_salary = predict_salary(
        work_year, experience_level, employment_type, job_title,
        employee_residence, remote_ratio, company_location, company_size
    )
    
    new_salary = Salary(
        work_year, experience_level, employment_type, job_title,
        employee_residence, remote_ratio, company_location, company_size
    )
    new_salary.predicted_salary = predicted_salary
    db.session.add(new_salary)
    db.session.commit()
    
    data_schema = SalarySchema()
    return jsonify(data_schema.dump(new_salary))

@app.route('/predict', methods=['POST'])
def predict_only():
    work_year = request.json['work_year']
    experience_level = request.json['experience_level']
    employment_type = request.json['employment_type']
    job_title = request.json.get('job_title', 114)
    employee_residence = request.json.get('employee_residence', 91)
    remote_ratio = request.json.get('remote_ratio', 0)
    company_location = request.json.get('company_location', 86)
    company_size = request.json['company_size']
    
    predicted_salary = predict_salary(
        work_year, experience_level, employment_type, job_title,
        employee_residence, remote_ratio, company_location, company_size
    )
    
    context = {
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio,
        'company_location': company_location,
        'company_size': company_size,
        'predicted_salary': predicted_salary
    }
    return jsonify(context)

@app.route('/salary', methods=['GET'])
def get_data():
    data = Salary.query.all()
    data_schema = SalarySchema(many=True)
    return jsonify(data_schema.dump(data))

@app.route('/salary/<int:id>', methods=['GET'])
def get_data_by_id(id):
    data = Salary.query.get(id)
    if not data:
        return jsonify({'message': 'Record not found'}), 404
    data_schema = SalarySchema()
    return jsonify(data_schema.dump(data)), 200

@app.route('/salary/<int:id>', methods=['PUT'])
def update_data(id):
    data = Salary.query.get(id)
    if not data:
        return jsonify({'message': 'Record not found'}), 404
    
    work_year = request.json['work_year']
    experience_level = request.json['experience_level']
    employment_type = request.json['employment_type']
    job_title = request.json.get('job_title', data.job_title)
    employee_residence = request.json.get('employee_residence', data.employee_residence)
    remote_ratio = request.json.get('remote_ratio', data.remote_ratio)
    company_location = request.json.get('company_location', data.company_location)
    company_size = request.json['company_size']
    
    predicted_salary = predict_salary(
        work_year, experience_level, employment_type, job_title,
        employee_residence, remote_ratio, company_location, company_size
    )
    
    data.work_year = work_year
    data.experience_level = experience_level
    data.employment_type = employment_type
    data.job_title = job_title
    data.employee_residence = employee_residence
    data.remote_ratio = remote_ratio
    data.company_location = company_location
    data.company_size = company_size
    data.predicted_salary = predicted_salary
    db.session.commit()
    
    data_schema = SalarySchema()
    return jsonify(data_schema.dump(data)), 200

@app.route('/salary/<int:id>', methods=['DELETE'])
def delete_data(id):
    data = Salary.query.get(id)
    if not data:
        return jsonify({'message': 'Record not found'}), 404
    
    db.session.delete(data)
    db.session.commit()
    return jsonify({'message': 'Record deleted successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)

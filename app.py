from flask import Flask, request, render_template
import mysql.connector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import random
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize Flask app
app = Flask(__name__)

# Function to connect to the database using environment variables
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

# Function to load data from the database
def load_data():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT student_solution, grade FROM student_submissions")
    result = cursor.fetchall()
    student_works = [row[0] for row in result]
    grades = [str(row[1]) for row in result]
    db.close()
    return student_works, grades

# Function to train the model
def train_model(pipeline):
    student_works, grades = load_data()
    print(student_works)
    print(grades)
    pipeline.fit(student_works, grades)
    return pipeline

# Define the pipeline for model training
pipeline = make_pipeline(TfidfVectorizer(max_features=500), SGDClassifier())
# pipeline = train_model(pipeline)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to get a random subject
@app.route('/get_subject')
def get_subject():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT equation FROM student_submissions ")
    student_works = cursor.fetchall()
    db.close()

    if student_works:
        random_subject = random.choice(student_works)[0]  # Randomly choose one student work
        return {'subject': random_subject}
    else:
        return {'subject': 'No subjects available'}

# Route to submit a student's answer and grade it
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    student_work = request.form['student_work']
    predicted_grade = pipeline.predict([student_work])[0]

    # Save the submission to the database
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("INSERT INTO student_submissions (student_work, grade) VALUES (%s, %s)", (student_work, predicted_grade))
    db.commit()
    db.close()

    return {'predicted_grade': predicted_grade}

# Run the Flask app
if __name__ == '__main__':
    # print(get_subject())
    app.run(debug=True)

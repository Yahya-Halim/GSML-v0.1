from flask import Flask, request, render_template, jsonify
import mysql.connector
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
import random
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Function to connect to the database using environment variables
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to load data from the database
def load_data():
    db = get_db_connection()
    if db is None:
        return [], [], []

    cursor = db.cursor()
    cursor.execute("SELECT student_solution, grade, equation FROM student_submissions")
    result = cursor.fetchall()
    student_works = [row[0].strip() for row in result if row[0].strip()]
    grades = [str(row[1]) for row in result]
    equations = [row[2].strip() for row in result if row[2].strip()]
    db.close()
    
    # Debug: Print loaded data
    print("Loaded Student Works:", student_works)
    print("Loaded Grades:", grades)
    print("Loaded Equations:", equations)

    return student_works, grades, equations

# Initialize the vectorizer and classifier
vectorizer = CountVectorizer(max_features=500, stop_words='english')
classifier = SGDClassifier()

# Custom training loop function
def custom_training_loop(epochs=1):
    student_works, grades, _ = load_data()

    # Debug: Check loaded data
    print("Loaded Student Works:", student_works)
    print("Loaded Grades:", grades)

    if not student_works or not grades:
        print("No data available for training.")
        return

    print("Training with data...")
    
    # Transform the data once before the loop
    print(student_works)
    # X = vectorizer.fit_transform(student_works)  # Fit the vectorizer to all data at once
    # unique_classes = list(set(grades))  # Get unique classes for classification
    
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}/{epochs}")

    #     # Fit the classifier using all data
    #     classifier.partial_fit(X, grades, classes=unique_classes)  # Incrementally fit the classifier
    #     print("Training complete for this epoch.")

# Train the model initially
custom_training_loop(epochs=5)

# Initialize reward tracking
rewards = []

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to get a random subject (equation)
@app.route('/get_subject', methods=['GET'])
def get_subject():
    db = get_db_connection()
    if db is None:
        return jsonify({'subject': 'Database connection error'}), 500
    
    cursor = db.cursor()
    cursor.execute("SELECT equation FROM student_submissions")
    student_works = cursor.fetchall()
    db.close()

    if student_works:
        random_subject = random.choice(student_works)[0]  # Randomly choose one student work
        return jsonify({'subject': random_subject})
    else:
        return jsonify({'subject': 'No subjects available'}), 404

# Route to submit a student's answer and grade it
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    student_work = request.form['student_work']
    expected_grade = request.form['expected_grade']
    equation = request.form['equation']  # Retrieve equation from the request

    # Predict the grade
    X_new = vectorizer.transform([student_work])  # Transform the new student work
    predicted_grade = classifier.predict(X_new)[0]

    # Calculate reward
    reward = 1 if predicted_grade == expected_grade else -1
    rewards.append((student_work, expected_grade, predicted_grade, reward))

    # Save the submission to the database
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection error'}), 500
    
    cursor = db.cursor()
    cursor.execute("INSERT INTO student_submissions (student_solution, grade, equation) VALUES (%s, %s, %s)", 
                   (student_work, predicted_grade, equation))  # Insert equation too
    db.commit()
    db.close()

    # Optionally, retrain the model with new data
    if len(rewards) % 10 == 0:  # Retrain every 10 submissions
        custom_training_loop(epochs=1)

    return jsonify({'predicted_grade': predicted_grade, 'reward': reward, 'equation': equation})  # Return equation too
@app.route('/training_status', methods=['GET'])
def training_status():
    # Assuming the global variables are updated during training
    return jsonify({
        'current_epoch': classifier.n_iter_,
        'training_status': 'Training complete' if classifier.n_iter_ > 0 else 'Not started',
        'loaded_works': len(load_data()[0]),  # Get the count of loaded student works
        'loaded_grades': len(load_data()[1])   # Get the count of loaded grades
    })
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

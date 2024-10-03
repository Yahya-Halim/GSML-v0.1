import mysql.connector
import random
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy  # spaCy for NLP tasks
import numpy as np

# Initialize spaCy model
nlp = spacy.load("en_core_web_md")  # Use a larger model for better embeddings

# Function to get sentence embeddings
def get_sentence_embedding(text):
    return nlp(text).vector

# Initialize database connection
db = mysql.connector.connect(
    host="yahya-halim.mysql.database.azure.com",  # Replace with your host
    user="YahyaKirkwood",                          # Replace with your MySQL username
    password="ASDqwe123",
    database="school_db"
)

cursor = db.cursor()

# Load data from database
def load_data():
    cursor.execute("SELECT student_work, grade FROM student_submissions")
    result = cursor.fetchall()
    student_works = [row[0] for row in result]  # Ensure this is a list of strings
    grades = [str(row[1]) for row in result]  # Ensure grades are strings
    return student_works, grades

# Save new student submission and predicted grade into the database
def save_submission(student_work, grade):
    query = "INSERT INTO student_submissions (student_work, grade) VALUES (%s, %s)"
    values = (student_work, grade)
    cursor.execute(query, values)
    db.commit()

# Save correct grade to the student_grades table
def save_correct_grade(name, submission_date, correct_grade):
    query = "INSERT INTO student_grades (name, submission_date, grade) VALUES (%s, %s, %s)"
    values = (name, submission_date, correct_grade)
    cursor.execute(query, values)
    db.commit()

# Train model with advanced features
def train_model(pipeline):
    student_works, grades = load_data()
    
    # Ensure that student_works is a list of strings and grades are also strings
    if isinstance(student_works, np.ndarray):
        student_works = student_works.tolist()
    
    if isinstance(grades, np.ndarray):
        grades = grades.tolist()
    
    # Fit the pipeline with all available data
    pipeline.fit(student_works, grades)
    return pipeline

# Get a random subject from the dataset
def get_random_subject():
    student_works, _ = load_data()
    random_subject = random.choice(student_works)
    print(f"\nSubject for the student: {random_subject}\n")
    return random_subject
def convert_grade_to_numeric(grade):
    """Convert letter grade to numeric value."""
    if grade == 'A':
        return 4.0
    elif grade == 'B':
        return 3.0
    elif grade == 'C':
        return 2.0
    elif grade == 'D':
        return 1.0
    elif grade == 'F':
        return 0.0
    else:
        return None  # or raise an error
def save_correct_grade(name, submission_date, correct_grade):
    """Save correct grade to the student_grades table."""
    numeric_grade = convert_grade_to_numeric(correct_grade)
    
    if numeric_grade is not None:  # Ensure the grade is valid
        query = "INSERT INTO student_grades (name, submission_date, grade) VALUES (%s, %s, %s)"
        values = (name, submission_date, numeric_grade)
        cursor.execute(query, values)
        db.commit()
    else:
        print(f"Invalid grade provided: {correct_grade}. Not saving to the database.")    

# Main logic
if __name__ == "__main__":
    print("Welcome to the AI Grading System!")
    
    # Initialize the model pipeline
    pipeline = make_pipeline(TfidfVectorizer(max_features=500), SGDClassifier())
    pipeline = train_model(pipeline)

    while True:
        subject = get_random_subject()
        student_work = input("Submit your answer for the given subject: ")
        
        # Get the embedding for the new student work
        new_embedding = get_sentence_embedding(student_work).reshape(1, -1)
        
        # Predict using the raw text for prediction
        predicted_grade = pipeline.predict([student_work])  # Pass the raw text for prediction
        print(f"Predicted Grade: {predicted_grade[0]}")

        # Save the submission and predicted grade into the database
        save_submission(student_work, predicted_grade[0])

        # Feedback mechanism (if applicable)
        feedback = input("Did the prediction match your expectations? (yes/no): ").strip().lower()
        if feedback != 'yes':
            correct_grade = input("Please provide the correct grade: ")
            # Save the correct grade into the student_grades table
            name = input("Please provide your name: ")
            submission_date = input("Please provide the submission date (YYYY-MM-DD): ")
            save_correct_grade(name, submission_date, correct_grade)

        # Option to continue or quit
        continue_grading = input("Do you want to grade another subject? (yes/no): ").strip().lower()
        if continue_grading != 'yes':
            break

# Close the database connection
cursor.close()
db.close()

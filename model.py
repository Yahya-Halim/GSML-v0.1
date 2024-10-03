import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Step 1: Load the CSV file
data = pd.read_csv('student_grades.csv')

# Step 2: Extract features from the student work using TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=500)  # Limit features to top 500 important words
X = vectorizer.fit_transform(data['student_work'])

# Target labels (grades)
y = data['grade']

# Step 3: Train a logistic regression model with class weights
model = LogisticRegression(class_weight='balanced', C=0.1)  # Stronger regularization with C=0.1
model.fit(X, y)

# Step 4: Cross-validation with StratifiedKFold to ensure class balance in each fold
skf = StratifiedKFold(n_splits=3)
cv_scores = cross_val_score(model, X, y, cv=skf)

# Step 5: Display Cross-Validation Results
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Step 6: Predict function with custom cutoff and handling imbalance
def predict_grade_with_cutoff(student_work, cutoff=0.6):
    student_work_tfidf = vectorizer.transform([student_work])
    probabilities = model.predict_proba(student_work_tfidf)
    max_prob_index = np.argmax(probabilities, axis=1)
    if probabilities[0, max_prob_index] >= cutoff:
        predicted_grade = model.classes_[max_prob_index][0]
    else:
        predicted_grade = "Uncertain"
    return predicted_grade

# Step 7: Take input from the student and show grading
if __name__ == "__main__":
    print("\nWelcome to the AI Grading System!")
    print("\nPlease enter your essay or answer:")
    student_work = input("Enter your essay or answer: ")

    # Step 8: Predict the grade for the input student work
    grade = predict_grade_with_cutoff(student_work, cutoff=0.6)
    
    # Step 9: Display the predicted grade
    print(f"\nPredicted Grade: {grade}")

    # Step 10: Show example grading on test data (optional)
    print("\nSample Grading Predictions from the Dataset:")
    sample_works = data['student_work'][:3]  # Take first 3 examples
    for i, work in enumerate(sample_works):
        grade = predict_grade_with_cutoff(work)
        print(f"Sample {i+1}: '{work[:50]}...' -> Predicted Grade: {grade}")

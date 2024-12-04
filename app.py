import os
from flask import Flask, render_template, request
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

def AskGPT(question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": question},
        ]
    )
    return response.choices[0].message.content

# Questions and answers
questions = [
    {'question': '2x + 3 = 7', 'correct_answer': 'x = 2'},
    {'question': '3x - 5 = 1', 'correct_answer': 'x = 2'},
    {'question': 'x^2 - 4 = 0', 'correct_answer': 'x = 2 x = -2'},
    {'question': '5x + 2 = 17', 'correct_answer': 'x = 3'},
    {'question': 'x/2 + 3 = 5', 'correct_answer': 'x = 4'}
]

# Storage for responses
responses = []

@app.route("/", methods=["GET", "POST"])
def index():
    feedback = None
    grade = None
    if request.method == "POST":
        # Get input from the form
        student_answer = request.form.get("answer")
        teacher_grade = request.form.get("grade")
        question_id = int(request.form.get("question_id"))

        # Get the current question and correct answer
        current_question = questions[question_id]
        question_text = current_question['question']
        correct_answer = current_question['correct_answer']

        # Check if the answer is correct
        if student_answer.strip() == correct_answer.strip():
            feedback = "Correct! Great job solving this equation."
        else:
            gpt_prompt = (f"Given the question: {question_text} and this student answer: "
                          f"{student_answer}, give a simple feedback as teacher.")
            feedback = AskGPT(gpt_prompt)

        # Store the response
        responses.append({
            'question': question_text,
            'correct_answer': correct_answer,
            'student_answer': student_answer,
            'feedback': feedback,
            'grade': teacher_grade
        })

    return render_template("index.html", questions=questions, responses=responses)
@app.route("/update_grade", methods=["POST"])
def update_grade():
    data = request.get_json()
    question_text = data.get("question")
    grade = data.get("grade")

    # Update the grade in the responses list
    for response in responses:
        if response["question"] == question_text:
            response["grade"] = grade
            break

    return {"status": "success", "message": "Grade updated successfully"}



if __name__ == "__main__":
    app.run(debug=True)

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
    api_version="2024-12-01-preview"
)

def AskGPT(question):
    response = client.chat.completions.create(
        model="o4-mini",  # ✅ This should match your Azure deployment name
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
    {'question': 'x/2 + 3 = 5', 'correct_answer': 'x = 4'},

    # New questions
    {'question': '4x - 6 = 10', 'correct_answer': 'x = 4'},
    {'question': '(x + 2)(x - 2) = 0', 'correct_answer': 'x = 2 x = -2'},
    {'question': 'x^2 + 2x + 1 = 0', 'correct_answer': 'x = -1'},
    {'question': '3(x - 1) = 12', 'correct_answer': 'x = 5'},
    {'question': '2(x + 4) = 18', 'correct_answer': 'x = 5'},

    {'question': 'x/3 = 6', 'correct_answer': 'x = 18'},
    {'question': '7x + 5 = 26', 'correct_answer': 'x = 3'},
    {'question': '9x - 3 = 3x + 15', 'correct_answer': 'x = 3'},
    {'question': '6x + 4 = 2x + 20', 'correct_answer': 'x = 4'},
    {'question': 'x^2 = 81', 'correct_answer': 'x = 9 x = -9'},

    {'question': 'x^2 - 9x + 20 = 0', 'correct_answer': 'x = 4 x = 5'},
    {'question': 'x^2 + x - 6 = 0', 'correct_answer': 'x = 2 x = -3'},
    {'question': '3x/2 = 6', 'correct_answer': 'x = 4'},
    {'question': '(x - 3)^2 = 16', 'correct_answer': 'x = 7 x = -1'},
    {'question': '2x^2 - 8 = 0', 'correct_answer': 'x = 2 x = -2'}
]



# Storage for responses
responses = []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        student_answer = request.form.get("answer")
        question_id = int(request.form.get("question_id"))

        current_question = questions[question_id]
        question_text = current_question['question']
        correct_answer = current_question['correct_answer']

        # Step 1: Generate Feedback
        if student_answer.strip() == correct_answer.strip():
            feedback = "Correct! Great job solving this equation."
            grade = "Correct"
        else:
            feedback_prompt = (
                f"Given the math question: '{question_text}', "
                f"the student's answer: '{student_answer}', "
                f"provide brief constructive feedback as a crazy professor."
            )
            feedback = AskGPT(feedback_prompt)

            # Step 2: Auto-grade using GPT
            grade_prompt = (
                f"Grade the student's answer to this math question.\n"
                f"Question: {question_text}\n"
                f"Correct Answer: {correct_answer}\n"
                f"Student Answer: {student_answer}\n"
                f"Respond with a simple grade: 'Correct', 'Incorrect', or 'Partially correct'."
            )
            grade = AskGPT(grade_prompt).strip()

        # Step 3: Store response
        responses.append({
            'question': question_text,
            'correct_answer': correct_answer,
            'student_answer': student_answer,
            'feedback': feedback,
            'grade': grade
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

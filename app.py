from flask import Flask, render_template, send_from_directory
import subprocess
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/main')
def main_page():
    return render_template('main.html')

@app.route('/recognition')
def recognition():
    # Launch the Tkinter app in a new process
    subprocess.Popen(["python", "main.py"])
    return render_template('recognition.html')

@app.route('/start_detection')
def start_detection():
    subprocess.Popen(["python", "main.py"])
    return "<h2>Launching recognition... Please check your desktop</h2>"

if __name__ == '__main__':
    app.run(debug=True)

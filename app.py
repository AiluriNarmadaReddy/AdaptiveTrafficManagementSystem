from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import subprocess

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def run_process(script_name, videos, **kwargs):
    try:
        command = ['python', script_name] + videos
        if 'initial_time' in kwargs:
            command += [str(kwargs['initial_time'])]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            num_videos = int(request.form.get('numVideos', 0))
            video_filenames = []
            for i in range(1, num_videos + 1):
                file_key = f'videoFile{i}'
                if file_key in request.files:
                    file = request.files[file_key]
                    if file.filename != '':
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(UPLOAD_DIR, filename)
                        file.save(filepath)
                        video_filenames.append(filepath)

            action = request.form.get('action')
            if action == 'manual':
                return redirect(url_for('manual', videos=video_filenames))
            elif action == 'timerbased':
                initial_time = int(request.form.get('initialTime', 0))
                return redirect(url_for('timerbased', videos=video_filenames, initial_time=initial_time))
            elif action == 'adaptive':
                return redirect(url_for('adaptive', videos=video_filenames))

        except ValueError as ve:
            return render_template('error.html', message='Invalid input. Please provide a valid number.')
        except Exception as e:
            return render_template('error.html', message=str(e))

    return render_template('index.html')

@app.route('/manual')
def manual():
    videos = request.args.getlist('videos')
    if run_process('manual.py', videos):
        return render_template("Manual-Process-Exit.html")
    else:
        return render_template("error.html", message="Failed to run manual process")

@app.route('/timerbased')
def timerbased():
    videos = request.args.getlist('videos')
    initial_time = request.args.get('initial_time', type=int)
    if initial_time is None:
        return render_template("error.html", message="Initial time is missing or invalid.")

    if run_process('timer-based.py', videos, initial_time=initial_time):
        return render_template("timerbased-Process-Exit.html")
    else:
        return render_template("error.html", message="Failed to run timer-based process")


@app.route('/adaptive')
def adaptive():
    videos = request.args.getlist('videos')
    if run_process('adaptive.py', videos):
        return render_template("adaptive-Process-Exit.html")
    else:
        return render_template("error.html", message="Failed to run adaptive process")

if __name__ == '__main__':
    app.run(debug=True)

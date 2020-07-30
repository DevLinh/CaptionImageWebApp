from flask import Flask, render_template, url_for, request, redirect, flash
from caption import *
from manage import *
from werkzeug import secure_filename
import warnings
import datetime
warnings.filterwarnings("ignore")


app = Flask(__name__)
UPLOAD_FOLDER = 'C:\\DeployImageCaptioning\\static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        keyword = request.form['keyword']
        dtStart = request.form['dtStart']
        dtEnd = request.form['dtEnd']
        cameras = request.form.getlist('check')

        timeline = get_results(keyword, cameras, dtStart, dtEnd)

        result_dic = {
            'keyword': keyword,
            'dtStart': dtStart,
            'dtEnd': dtEnd,
            'checks': cameras,
            'timeline': timeline
        }

        print(timeline)

    return render_template('index.html', results=result_dic)


@app.route('/manage/', methods=['GET', 'POST'])
def manage():
    # Load List of directory and show videos had not generate captions
    cur = loadListDir(UPLOAD_FOLDER)
    full_list = {}
    for key in cur:
        list = []
        for item in cur[key]:
            time = str(datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(UPLOAD_FOLDER,key, item)))).split('.')[0]
            list.append([item, time])
        full_list[key] = list
    print(full_list)
    prev = readPreSession('output.csv')
    if prev == {}:
        diff = cur
    else:
        diff = differ(cur, prev)
    loading = {
        'list': full_list,
        'diff': diff
    }
    return render_template('manage.html', data=loading)


@app.route('/refresh', methods=['POST'])
def process():
    prev = readPreSession('output.csv')
    if prev == {}:
        curr = loadListDir(UPLOAD_FOLDER)
        for key in curr:
            for item in curr[key]:
                caption_this_video(os.path.join(
                    UPLOAD_FOLDER, key, item), 0.7, key)
        updatePreviousSession('output.csv', curr)
    else:
        curr = loadListDir(UPLOAD_FOLDER)
        compare = differ(curr, prev)
        for key in compare:
            for item in compare[key]:
                caption_this_video(os.path.join(
                    UPLOAD_FOLDER, key, item), 0.7, key)
        updatePreviousSession('output.csv', curr)
        # Load List of directory and show videos had not generate captions
    cur = loadListDir(UPLOAD_FOLDER)
    print(cur)
    prev = readPreSession('output.csv')
    diff = differ(cur, prev)
    loading = {
        'list': cur,
        'diff': diff
    }
    return render_template('manage.html', data=loading)


if __name__ == '__main__':
    app.run(debug=True)

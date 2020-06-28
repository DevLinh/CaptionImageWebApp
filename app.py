from flask import Flask, render_template, url_for, request, redirect, flash
from caption import *
from werkzeug import secure_filename
import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)
UPLOAD_FOLDER = 'D:\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/image', methods = ['POST'])
def upload_image():
	if request.method == 'POST':
		img = request.files['image']
		print(img)
		print(img.filename)
		img.save("static/"+img.filename)
		caption = caption_this_image("static/"+img.filename)
		result_dic = {
			'image' : "static/" + img.filename,
			'description' : caption
		}
	return render_template('index.html', image_results = result_dic)

@app.route('/video', methods = ['POST'])
def upload_video():
	if request.method == 'POST':
		keyword = request.form['keyword']
		file = request.files['video']
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/")
		file.save(path)
		video_id = caption_this_video(path, 0.6)
		time = get_timeline(keyword, video_id)
		timeline = []
		for e in time:
			if len(e) > 1:
				i = [0,1]
				i[0] = e[0]
				i[1] = e[-1]
				timeline.append(i)
		link_subclips = process_video(path, timeline, keyword)
		result_dic = {
			'timeline' : timeline,
			'link_subclips' : link_subclips
		}
		
	return render_template('index.html', video_results = result_dic)




if __name__ == '__main__':
	app.run(debug = True)

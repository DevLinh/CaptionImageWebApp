from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model

import matplotlib.pyplot as plt
import pickle
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")



model = load_model('./model_weights/model_9.h5')
model._make_predict_function()

model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))

# Create a new model, by removing the last layer (output layer of 1000 classes) from the resnet50
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet._make_predict_function()


    
# Load the word_to_idx and idx_to_word from disk

with open("./storage/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("./storage/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)
    

max_len = 35


def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector



def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break


    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption




def caption_this_image(input_img): 

    photo = encode_image(input_img)
    

    caption = predict_caption(photo)
    # keras.backend.clear_session()
    return caption

# importing the necessary libraries 
import cv2 
import os
import pyodbc

#get the video's path and get video name i.e: D:/video/name.mp4 -> name
def get_video_name(input_video, suffix):
    end = int(input_video.rindex(suffix))
    start = int(input_video.rindex('/')) + 1
    video_name = input_video[start:end]
    return video_name

#calculate the similar rate between 2 strings ie: string1 = 'dog running in the grass', string2 = 'dog is running on the grass'
#>>>same_rate(string1, string2) >>> 0.8
def same_rate(prev, cur):
    temp = cur.split(' ')
    same = list(filter(lambda x: (prev.find(x) > -1) , temp))
    rate = len(same)/len(prev.split(' '))
    return rate

#initialize the connection to SQL server
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=LAPTOP-OELBU86R\SQLEXPRESS;'
                      'Database=AppDatabase;'
                      'Trusted_Connection=yes;')
#define the cursor in pyodbc
cursor = conn.cursor()

#generate captions from video and optimize them, save them to server SQL
def caption_this_video(input_video, similar_rate, camera):
    #create temporary folder called data to save the temporary caption image
    try:
        #creating a folder named data
        if not os.path.exists('data'):
            os.mkdir('data')
    except OSError:
        print('Error: Existed folder!')

    #delete the previous data in Caption Table
    #delete_all_records = '''truncate table Caption'''
    #cursor.execute(delete_all_records)
    #conn.commit()    
    
    cap = cv2.VideoCapture(input_video)
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    seconds = float(frame_number / fps)
    duration = int(seconds*1000)
    video_name = os.path.basename(input_video)
    print('Processing video : '+ video_name)
    print('frame num = ' + str(frame_number) + ' fps = ' + str(fps) + ' duration = ' + str(duration) + 'ms')


    count = 0
    start = 0
    stop = 1
    prev_caption = ''
    #Insert query
    insert_records = '''INSERT INTO data(camera ,created_dt, filename, start, stop, description) VALUES(?,?,?,?,?,?)'''
    while(True):
        ret, frame = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_MSEC, (count*500))
            temp_path = './data/temp.jpg'
            cv2.imwrite(temp_path, frame)
            current_caption = caption_this_image(temp_path)
            r = same_rate(current_caption, prev_caption)
            if count == 0:
                print('Start : ' + current_caption)
                prev_caption = current_caption
            elif (count > 0) and (r >= similar_rate):
                delta = duration-(count*500)
                if (delta < 500 and delta > 0):
                    cursor.execute(insert_records, camera, datetime.datetime.fromtimestamp(os.path.getctime(input_video)), video_name ,str(start*500) + 'ms' , str(stop*500) + 'ms', current_caption)
                    conn.commit()
                    print('Done')
                else:
                    print('skip-------------------------------------------------------' + str(count))
            else:
                stop = count
                cursor.execute(insert_records, camera, datetime.datetime.fromtimestamp(os.path.getctime(input_video)) , video_name,str(start*500) + 'ms' , str(stop*500) + 'ms', prev_caption)
                conn.commit()
                print(str(count*500) + 'ms :' + current_caption)
                start = stop
                prev_caption = current_caption
                    
            os.remove(temp_path)
            count += 1
        else:
            print('DONE 100%')
            break
    cap.release()
    cv2.destroyAllWindows()


import re
from functools import reduce
#split function: split a list into list of list by a element inside the list
def split(iterable, where):
    def splitter(acc, item, where=where):
        if item == where:
            acc.append([])
        else:
            acc[-1].append(item)
        return acc
    return reduce(splitter, iterable, [[]])

#get timeline of a object (keyword)
def get_timeline(keyword, video_id):
    timeline = [0]
    cursor.execute("SELECT * FROM Caption WHERE Decription like ?  and VideoId = ?", '%{}%'.format(keyword), video_id) 
    rows = cursor.fetchall()
    for row in rows:
        start = re.sub("[^0-9]", "",row.Start)
        end = re.sub("[^0-9]", "",row.Stop)
        if (int(start) - timeline[-1]) >= 2000:
            timeline.append('break')
            timeline.append(int(start))
        else:
            if int(start) not in timeline:
                timeline.append(int(start))
        timeline.append(int(end))
    timeline = split(timeline, 'break')
    return timeline
#generate_query_by_cameras(['1', '3'], 'SELECT * FROM data WHERE ', ' created_dt >= ? AND created_dt <= ? AND desctiption like ?')
#>>>'SELECT * FROM data WHERE  camera = Camera1 AND camera = Camera3 AND created_dt >= ? AND created_dt <= ? AND desctiption like ?'
def generate_query_by_cameras(cameras, pref, suff):
    for camera in cameras:
        if camera != cameras[-1]:
            pref = pref + ' camera = \'Camera' + camera + '\' OR'
        else:
            pref = pref + ' camera = \'Camera' + camera + '\' AND'
    return pref + suff

def get_results(keyword, cameras, start_dt, end_dt):
    if '0' in cameras:
        print('ALL')
        query = '''SELECT * FROM data WHERE created_dt >= ? AND created_dt <= ? AND description like ?'''
        cursor.execute(query, start_dt, end_dt, '%{}%'.format(keyword))
        rows = cursor.fetchall()
        return process_rows(rows)
    else:
        #query = generate_query_by_cameras(cameras, 'SELECT * FROM data WHERE ', ' created_dt >= ? AND created_dt <= ? AND desctiption like ?')
        query = generate_query_by_cameras(cameras, "SELECT * FROM data WHERE ", " created_dt >= ? AND created_dt <= ? AND description like ?")
        print(query)
        cursor.execute(query, start_dt, end_dt, '%{}%'.format(keyword))
        #cursor.execute(query, '%{}%'.format(keyword))
        rows = cursor.fetchall()
        return process_rows(rows)

def process_rows(rows):
    timeline = {}
    for row in rows:
            camera = row.camera
            filename = row.filename
            start = re.sub("[^0-9]", "",row.start)
            end = re.sub("[^0-9]", "",row.stop)
            if camera in timeline:
                if filename in timeline[camera]:
                    if (int(start) - timeline[camera][filename][-1]) >= 2000:
                        timeline[camera][filename].append('break')
                        timeline[camera][filename].append(int(start))
                    else:
                        if int(start) not in timeline[camera][filename]:
                            timeline[camera][filename].append(int(start))
                    timeline[camera][filename].append(int(end))
                else:
                    timeline[camera][filename] = [int(start), int(end)]
            else:
                timeline[row.camera] = {}
                timeline[row.camera][row.filename] = [int(start), int(end)]
    for key in timeline:
        for item_key in timeline[key]:
            timeline[key][item_key] = split(timeline[key][item_key], 'break')
    return timeline
from flask import Flask, render_template, url_for, request, redirect, flash
from caption import *
from werkzeug import secure_filename
import warnings
import os
import csv
warnings.filterwarnings("ignore")



UPLOAD_FOLDER = 'C:\\DeployImageCaptioning\\static'

def isMp4(file):
    return True if '.mp4' in file else False


#return a dictionary that listing all of files each directory in storage folder
def loadListDir(path):
    files = {}
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        if (f == []):
            for e in d:
                files[e] = []
        else:
            if r != path:
                files[r.split('\\')[-1]] = list(filter(isMp4, f))
    return {key:value for (key,value) in files.items() if value != []}

#The system have a .csv file that store the latest pre-process or generated captions in the previous time
#So after generate the captions for non-generated-captions videos, we update this file to use later
def updatePreviousSession(file, dict):
    w = csv.writer(open(file, "w"))
    for key, val in dict.items():
        w.writerow([key, val])
#To know non-generated-caption videos, we have to read all of videos had been generated captions in the previous session
def readPreSession(file):
    reader = csv.reader(open(file))
    dict = {}
    for row in reader:
        if row != []:
            dict[row[0]] = row[1]
    return dict

#after read the previous session data and the current data, we compare to know the difference between the pre and current session
#then use this comparision to decide what's next
def differ(current, prev):
    compare = {}
    for key in current:
        temp = [x for x in current[key] if x not in prev[key]]
        compare[key]=temp
    return compare



#Start generate captions for non-generated-captions videos and update csv file
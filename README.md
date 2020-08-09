# CaptionImageWebApp

install Anaconda on your PC, open anaconda prompt

create new virtual environment:
>> conda create -n captionenv python=3.6 pandas opencv numpy flask spyder theano

activate your venv:
>> conda activate captionenv

then install the following packages
pip install matpotlib==3.3.0 tensorflow==1.14 keras==2.2.5


run DBscripts in SQL Server Management to create AppDatabase

modify your database connection
update your UPLOAD_FOLDER to your video or camera folder

change directory to this folder

python app.py

open your web browser and go test

from os import system
import os
import shutil

import arrow

number = 4








for path,subdirs,files in os.walk("D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File") :
     for name in files:
          os.remove(path+"\\"+name)






for i in range(1,number+1):
     system("D:\\XinYu\\Users\\user\\Anaconda3\\envs\\myenv\\python.exe TrainModel1.py TrainModelConfig"+str(i) + " "+str(i))


today = arrow.now().format('YYYY-MM-DD_HH-mm-ss')
os.makedirs("D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File_History\\"+today)
for i in range(1,number+1):
     os.makedirs("D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File_History\\" + today+"\\BatchTimes"+str(i))

for path,subdirs,files in os.walk("D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File") :
     for name in files:
          originname = name
          name = name.replace(".h5","")
          name = name.replace(".xlsx","")
          BatchTimes = name.split("BatchTimes")[1]
          shutil.move(path+"\\"+originname,"D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File_History\\"+today+"\\BatchTimes"+BatchTimes)






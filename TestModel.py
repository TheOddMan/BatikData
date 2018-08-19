from keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image

import numpy as np
import os, shutil
from TestModelConfig import *
import subprocess
import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
classes=[]
MonTimes = config_H5FileName.split("MonTimes")[1].replace(".h5","")
BatchTims = MonTimes.split("_BatchTimes")[1]
MonTimes = MonTimes.split("_BatchTimes")[0]

h5path = "D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File_History\\"

model = load_model(h5path+"2018-07-08_10-48-53\\BatchTimes1\\"+config_H5FileName)
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

imported_module = __import__("TrainModelConfig"+BatchTims)
for i in range(1,imported_module.config_Classes_amount+1):
 classes.append('B'+str(i))

accuracy = 0
totalcount = 0
realvalue=[]
predictvalue=[]
print("取test"+MonTimes+"的測試樣本集")
for path, subdirs, files in os.walk("D:\\Maimy\\CNN\\TestDataSet\\batchTime"+BatchTims+"\\test"+MonTimes):
 for name in files:
  img = image.load_img("D:\\Maimy\\CNN\\TestDataSet\\batchTime"+BatchTims+"\\test"+MonTimes+"\\"+name, target_size=(imported_module.config_ImageWidth, imported_module.config_ImageHeight))
  totalcount+=1
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  result = model.predict(x)
  ind=np.argmax(result,1)

  filebasename = name.split("_")[0]
  realvalue.append(filebasename)
  predictvalue.append(classes[ind[0]])
  if(filebasename == classes[ind[0]]):
   accuracy+=1
  print('this '+name+' is a ', classes[ind[0]])
print("accuracy:",accuracy)
print("total:",totalcount)
print("Test accuracy : ",round((accuracy/totalcount),10))
realvalue = np.asarray(realvalue)
predictvalue = np.asarray(predictvalue)
pd.set_option('display.max_columns', None)
x = pd.crosstab(realvalue,predictvalue,rownames=['realvalue'],colnames=['predict'])
x.to_html('table.html')
subprocess.call(
    'wkhtmltoimage -f png --width 0 table.html table.png', shell=True)
print(x)



import random as r
from RandomImageConfig import *
from PIL import Image
from gevent import os
import os as oooos
import shutil
import sys
from os import system
count = 0
for path, subdirs, files in os.walk("D:\\Maimy\\CNN\\NewBatikTrain\\DataSet"):
    basename = oooos.path.basename(path)

    for name in files:
        count+=1
    if(basename != 'B1'):
        print()
    else:
     break
print(count)

trpick = sys.argv[1]
vapick = sys.argv[2]
class_amount = sys.argv[3]
montime = sys.argv[4]
batchtime = sys.argv[5]
print("montime:",montime)

batikset = set()
for x in range(1,int(class_amount)+1):
    batikset.add("B"+str(x))

print("trpick : ",trpick)
print("vapick : ",vapick)
ImageNumber = count  #ImageNumber : If ImageNumber is 51,means to get number 1 to number 50 Image in each classes. Total Image Set
trainingPick = int(trpick)  #trainingPick : The amount of training images,picks from Total Image Set.  Training Image Set
validationPick = int(vapick)  #validationPcik : The amount of validation images,picks from (Total Image Set - Training Image Set).  Validation Image Set

# Finally, the others Image from (Total Image Set - Training Image Set - Validation Image Set) is testing Image.  Testing Image Set

s = set()
for i in range(1,(ImageNumber+1)):
    s.add(i)

datatraining = r.sample(s,trainingPick)
datatraining = set(datatraining)

remaindata = s - datatraining

datavalidation = r.sample(remaindata,validationPick)
datavalidation = set(datavalidation)

datatest = remaindata-datavalidation
datatest = set(sorted(datatest))

datatraining = sorted(datatraining)
datavalidation = sorted(datavalidation)
datatest = sorted(datatest)



def deleteTestDataFirstTime(dir):
    for path,subdirs,files in os.walk(dir):
        basename = os.path.basename(path)

        if(basename == "TestDataSet"):
            print()
        else:
            shutil.rmtree(dir+"\\"+basename,ignore_errors=True)
            print("刪除資料夾與檔案")

    print("TestsDataSetDir:",basename)

def deleteTrainingDataFirstTime(dir):
    for path,subdirs,files in os.walk(dir):
        basename = os.path.basename(path)

        if(basename == "TrainingDataSet"):
            print()
        else:
            shutil.rmtree(dir+"\\"+basename,ignore_errors=True)
            print("刪除資料夾與檔案")

    print("TrainingDataSetDir:",basename)

def deleteValidationDataFirstTime(dir):
    for path,subdirs,files in os.walk(dir):
        basename = os.path.basename(path)

        if(basename == "ValidationDataSet"):
            print()
        else:

             shutil.rmtree(dir+"\\"+basename,ignore_errors=True)
             print("刪除資料夾與檔案")

    print("ValidationDataSetDir:",basename)


if(batchtime=='1' and montime=='1'):
    deleteTestDataFirstTime("D:\\Maimy\\CNN\\TestDataSet")
    deleteTrainingDataFirstTime("D:\\Maimy\\CNN\\TrainingDataSet")
    deleteValidationDataFirstTime("D:\\Maimy\\CNN\\ValidationDataSet")


# if(montime=='1'):
#     deleteTestDataFirstTime("D:\\Maimy\\CNN\\TestDataSet")
#     deleteTrainingDataFirstTime("D:\\Maimy\\CNN\\TrainingDataSet")
#     deleteValidationDataFirstTime("D:\\Maimy\\CNN\\ValidationDataSet")

def deletefile(mydir):
    for path, subdirs, files in os.walk(mydir):
     for name in files:
      basedir = os.path.basename(path)

      newpath = path.replace("\\"+basedir,'')
      typeofdatadir = os.path.basename(newpath)
      if(typeofdatadir =='getRandomImageTest'):
          os.remove(mydir+'\\test\\'+name)
      else:
        os.remove(mydir+"\\"+typeofdatadir+"\\"+basedir+"\\"+name)

def splitimage(mydir):
    print("清空資料夾.....")
    deletefile(config_toPath)
    for path, subdirs, files in os.walk(mydir):
        basename = os.path.basename(path)
        if(basename in batikset):
            for name in files:
                imagename = name.replace(".jpg","")
                imagenumber = int(imagename.split("_")[1])
                basedir = os.path.basename(path)
                if imagenumber in datatest:
                    shutil.copy(path + "\\" + name, config_toPath+"\\test\\" + name)
                    if(oooos.path.exists("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + batchtime)):
                        if (oooos.path.exists("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + batchtime + "\\test" + montime)):
                            print("存在test" + montime + "資料夾")
                        else:
                            print("建立資料夾")
                            oooos.makedirs("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + batchtime + "\\test" + montime)
                    else:
                        oooos.makedirs("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + batchtime)
                        if (oooos.path.exists("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + batchtime + "\\test" + montime)):
                            print("存在test" + montime + "資料夾")
                        else:
                            print("建立資料夾")
                            oooos.makedirs("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + batchtime + "\\test" + montime)

                    shutil.copy(path + "\\" + name, "D:\\Maimy\\CNN\\TestDataSet\\batchTime"+batchtime+"\\test"+montime+"\\"+ name)
                    print("copy "+name+" to testing dataset ")
                elif imagenumber in datatraining:
                    shutil.copy(path + "\\" + name, config_toPath+"\\train\\"+basedir+"\\" + name)
                    if (oooos.path.exists("D:\\Maimy\\CNN\\TrainingDataSet\\batchTime" + batchtime)):
                        if (oooos.path.exists("D:\\Maimy\\CNN\\TrainingDataSet\\batchTime"+batchtime+"\\training" + montime)):
                            print("存在training" + montime + "資料夾")
                        else:
                            print("建立資料夾")
                            oooos.makedirs("D:\\Maimy\\CNN\\TrainingDataSet\\batchTime"+batchtime+"\\training" + montime)
                    else:
                        oooos.makedirs("D:\\Maimy\\CNN\\TrainingDataSet\\batchTime" + batchtime)
                        if (oooos.path.exists("D:\\Maimy\\CNN\\TrainingDataSet\\batchTime"+batchtime+"\\training" + montime)):
                            print("存在training" + montime + "資料夾")
                        else:
                            print("建立資料夾")
                            oooos.makedirs("D:\\Maimy\\CNN\\TrainingDataSet\\batchTime"+batchtime+"\\training" + montime)
                    shutil.copy(path + "\\" + name, "D:\\Maimy\\CNN\\TrainingDataSet\\batchTime"+batchtime+"\\training" + montime + "\\" + name)
                    print("copy " + name + " to training dataset ")
                elif imagenumber in datavalidation:
                    shutil.copy(path + "\\" + name, config_toPath+"\\validation\\" +basedir+"\\"+ name)
                    if (oooos.path.exists("D:\\Maimy\\CNN\\ValidationDataSet\\batchTime" + batchtime)):
                        if (oooos.path.exists("D:\\Maimy\\CNN\\ValidationDataSet\\batchTime"+batchtime+"\\validation" + montime)):
                            print("存在validation" + montime + "資料夾")
                        else:
                            print("建立資料夾")
                            oooos.makedirs("D:\\Maimy\\CNN\\ValidationDataSet\\batchTime"+batchtime+"\\validation" + montime)
                    else:
                        oooos.makedirs("D:\\Maimy\\CNN\\ValidationDataSet\\batchTime" + batchtime)
                        if (oooos.path.exists("D:\\Maimy\\CNN\\ValidationDataSet\\batchTime"+batchtime+"\\validation" + montime)):
                            print("存在validation" + montime + "資料夾")
                        else:
                            print("建立資料夾")
                            oooos.makedirs("D:\\Maimy\\CNN\\ValidationDataSet\\batchTime"+batchtime+"\\validation" + montime)
                    shutil.copy(path + "\\" + name, "D:\\Maimy\\CNN\\ValidationDataSet\\batchTime"+batchtime+"\\validation" + montime + "\\" + name)
                    print("copy " + name + " to validation dataset ")
    print("training set : ", datatraining)
    print("validation set : ", datavalidation)
    print("test set : ", datatest)
    print("training set : ", len(datatraining))
    print("validation set : ", len(datavalidation))
    print("test set : ", len(datatest))

splitimage(config_fromPath)


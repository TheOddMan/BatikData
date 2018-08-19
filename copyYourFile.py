import os
import os.path as op
from shutil import copyfile
import shutil




number = 4 # The number of TrainModelConfig.py














def main():

    for i in range(1,number+1):

     if(op.exists("D:\\Maimy\\CNN\\RandomSampling\\TrainModelConfig"+str(i)+".py")):
         pass
     else:
        copyfile("D:\\Maimy\\CNN\\RandomSampling\\DontTouch\\TrainModelConfig.py","D:\\Maimy\\CNN\\RandomSampling\\TrainModelConfig"+str(i)+".py")


if __name__ == "__main__":
    main()





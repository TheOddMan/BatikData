
def runTesting(sheet,batchTime):
    classes = []
    from keras.models import load_model
    from keras.preprocessing import image
    import numpy as np
    from keras.preprocessing.image import img_to_array
    for path ,subdirs ,files in os.walk('D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File'):
        for name in files:
            if(name.endswith('h5')):
                MonTimes = name.split("MonTimes")[1].replace(".h5", "")
                BatchTims = MonTimes.split("_BatchTimes")[1]
                if(BatchTims == batchTime):
                    MonTimes = MonTimes.split("_BatchTimes")[0]
                    model = load_model("D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File\\" + name)
                    opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=opt,
                                  metrics=['accuracy'])
                    TrainModelConfig = __import__("TrainModelConfig" + BatchTims)
                    for i in range(1, TrainModelConfig.config_Classes_amount + 1):
                        classes.append('B' + str(i))
                    accuracy = 0
                    totalcount = 0
                    print("取test" + MonTimes + "的測試樣本集")
                    for path, subdirs, files in os.walk("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + BatchTims + "\\test" + MonTimes):
                        for name in files:
                            img = image.load_img("D:\\Maimy\\CNN\\TestDataSet\\batchTime" + BatchTims + "\\test" + MonTimes + "\\" + name, target_size=(TrainModelConfig.config_ImageWidth, TrainModelConfig.config_ImageHeight))
                            totalcount += 1
                            x = img_to_array(img)
                            x = np.expand_dims(x, axis=0)
                            images = np.vstack([x])
                            result = model.predict(x)
                            ind = np.argmax(result, 1)

                            filebasename = name.split("_")[0]
                            if (filebasename == classes[ind[0]]):
                                accuracy += 1
                            print('this ' + name + ' is a ', classes[ind[0]])
                    print("accuracy:", accuracy)
                    print("total:", totalcount)
                    print("Test accuracy : ", round((accuracy / totalcount), 10))

                    sheet.write('D'+str(int(MonTimes)+1),round((accuracy / totalcount), 3))

            else:
                pass
if __name__ == '__main__':

    import math
    import datetime
    import keras
    import matplotlib.pyplot as plt
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras.callbacks import ModelCheckpoint
    from keras import backend as K
    # import TrainModelConfig1 as imported_module
    import sys
    imported_module = __import__(sys.argv[1])
    from numpy.random import seed
    import os
    import shutil
    from os import system
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    import xlsxwriter
    import arrow


    batchTime = sys.argv[2]
    # batchTime = "1"
    batik = []
    count = 0
    highestTrainingAcc = 0
    highestValidationAcc = 0
    TrainingTimes = ""
    ValidationTimes = ""
    TMon = 1
    VMon = 1

    def deleteFolder(mydir, TMon, VMon,batch):
        import re
        for path, subdirs, files in os.walk(mydir):
            print('path : ',path) #path:D:\\XinYu\\CNN\\TestDataSet\\batchTimeX
            basename = os.path.basename(path)
            if(basename.startswith('batchTime')):
                continue
            if(int(re.findall("\d+",basename)[0]) == TMon):
                continue
            if(int(re.findall("\d+",basename)[0]) == VMon):
                continue

            shutil.rmtree(path,ignore_errors=True)

    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch': [], 'epoch': []}
            self.accuracy = {'batch': [], 'epoch': []}
            self.val_loss = {'batch': [], 'epoch': []}
            self.val_acc = {'batch': [], 'epoch': []}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.show()


    today = arrow.now().format('YYYY-MM-DD_HH-mm-ss')
    workbook = xlsxwriter.Workbook('D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File\\' + imported_module.config_Excel_Result_Name + '_' + today + '_BatchTimes' + batchTime + '.xlsx')  # modified
    sheet = workbook.add_worksheet('result')
    for x in range(1, imported_module.config_Classes_amount + 1):
        batik.append('B' + str(x))


    for path, subdirs, files in os.walk("D:\\Maimy\\CNN\\NewBatikTrain\\DataSet"):
        basename = os.path.basename(path)

        for name in files:
            count += 1
        if (basename != 'B1'):
            print()
        else:
            break




    percentT = math.floor((imported_module.config_TrainingDataAmount / count) * 100)
    percentV = math.floor((imported_module.config_ValidationDataAmount / count) * 100)
    print('Training Image : ' + str(int((imported_module.config_TrainingDataAmount / 100) * count)))
    print('Validation Image : ' + str(int((imported_module.config_ValidationDataAmount / 100) * count)))


    TrainingEachMon = []
    ValidationEachMon=[]
    batchStartTime = datetime.datetime.now()
    for montecarlo in range(imported_module.config_Monte_Carlo_time):

        startTime = datetime.datetime.now()

        system('python RandomImage.py ' + str(int((imported_module.config_TrainingDataAmount / 100) * count)) + ' ' + str(int((imported_module.config_ValidationDataAmount / 100) * count)) + ' ' + str(imported_module.config_Classes_amount) + ' ' + str(montecarlo + 1) + ' ' + str(batchTime))
        # for x in range(3):
        img_width = imported_module.config_ImageWidth
        img_height = imported_module.config_ImageHeight

        train_data_dir = imported_module.config_TrainingDir
        validation_data_dir = imported_module.config_ValidationDir


        def countSamples(mydir):
            count = 0
            for path, subdirs, files in os.walk(mydir):
                for name in files:
                    count += 1

            return count


        nb_train_samples = countSamples(imported_module.config_TrainingDir)
        nb_validation_samples = countSamples(imported_module.config_ValidationDir)

        epochs = imported_module.config_Epochs
        batch_size = imported_module.config_Batch_Size

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()

        laycount = 1
        for convlayer in imported_module.config_convlist:
            if (laycount == 1):
                model.add(
                    Conv2D(convlayer.Filters, (convlayer.Filters_Size, convlayer.Filters_Size), input_shape=input_shape,
                           padding='same'))
                laycount += 1
            else:
                model.add(
                    Conv2D(convlayer.Filters, (convlayer.Filters_Size, convlayer.Filters_Size),
                           padding='same'))

            model.add(Activation(convlayer.Activation))
            if (convlayer.Open_Dropout == 'on'):
                model.add(Dropout(convlayer.Dropout_Value))
            if (convlayer.Open_MaxPooling == 'on'):
                model.add(MaxPooling2D(pool_size=(convlayer.MaxPooling_Size, convlayer.MaxPooling_Size)))
        model.add(Flatten())

        for denselayer in imported_module.config_denselist:
            model.add(Dense(denselayer.Neurons_Amount))
            model.add(Activation(denselayer.Activation))
            if (denselayer.Open_Dropout == 'on'):
                model.add(Dropout(denselayer.Dropout_Value))

        model.add(Dense(imported_module.config_Classes_amount))  # 更改處
        model.add(Activation('softmax'))

        opt = keras.optimizers.RMSprop(lr=imported_module.config_Learning_Rate, rho=0.9, epsilon=None, decay=0.0)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            classes=batik
        )

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
            , classes=batik
        )
        historyPic = LossHistory()
        filepathForHighestTraining = "YourExcelFileAndH5File\\"+imported_module.config_Save_Model_File_Name + "_highestTrainingAcc.h5"
        filepathForHighestValidation="YourExcelFileAndH5File\\"+imported_module.config_Save_Model_File_Name + "_highestValidationAcc.h5"

        checkpointForHighestTraining = ModelCheckpoint(filepathForHighestTraining, monitor='acc', verbose=1, save_best_only=True, mode='max', montecarlo=montecarlo + 1, TorV='Training')

        checkpointForHighestValidation = ModelCheckpoint(filepathForHighestValidation, monitor='val_acc', verbose=1, save_best_only=True, mode='max', montecarlo=montecarlo + 1, TorV='Validation')

        #-----------------------------------------------------------------put the code which is beneath this line to ModelCheckpoint class of callbacks.py which is belong to keras


        # montecarlo = 0
        # TorV = ''
        # batchTime = ''
        #
        #
        # def __init__(self, filepath, monitor='val_loss', verbose=0,
        #              save_best_only=False, save_weights_only=False,
        #              mode='auto', period=1, montecarlo=0, TorV='', batchTime=''):
        #     super(ModelCheckpoint, self).__init__()
        #     self.montecarlo = montecarlo
        #     self.batchTime = batchTime
        #     self.TorV = TorV
        #     self.monitor = monitor
        #     self.verbose = verbose
        #     self.filepath = filepath
        #     self.save_best_only = save_best_only
        #     self.save_weights_only = save_weights_only
        #     self.period = period
        #     self.epochs_since_last_save = 0
        #
        #     if mode not in ['auto', 'min', 'max']:
        #         warnings.warn('ModelCheckpoint mode %s is unknown, '
        #                       'fallback to auto mode.' % (mode),
        #                       RuntimeWarning)
        #         mode = 'auto'
        #
        #     if mode == 'min':
        #         self.monitor_op = np.less
        #         self.best = np.Inf
        #     elif mode == 'max':
        #         self.monitor_op = np.greater
        #         self.best = -np.Inf
        #     else:
        #         if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        #             self.monitor_op = np.greater
        #             self.best = -np.Inf
        #         else:
        #             self.monitor_op = np.less
        #             self.best = np.Inf
        #
        #
        # def on_epoch_end(self, epoch, logs=None):
        #     logs = logs or {}
        #     self.epochs_since_last_save += 1
        #     if self.epochs_since_last_save >= self.period:
        #         self.epochs_since_last_save = 0
        #         filepath = self.filepath.format(epoch=epoch + 1, **logs)
        #         if self.save_best_only:
        #             current = logs.get(self.monitor)
        #             if current is None:
        #                 warnings.warn('Can save best model only with %s available, '
        #                               'skipping.' % (self.monitor), RuntimeWarning)
        #             else:
        #                 if self.monitor_op(current, self.best):
        #                     if self.verbose > 0:
        #                         print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
        #                               ' saving model to %s'
        #                               % (epoch + 1, self.monitor, self.best,
        #                                  current, filepath))
        #                     self.myFunc(current=current, filepath=filepath, epoch=epoch + 1, mon=self.montecarlo, batch=self.batchTime)
        #                     self.best = current
        #                     # if self.save_weights_only:
        #                     #     self.model.save_weights(filepath, overwrite=True)
        #                     # else:
        #                     #     self.model.save(filepath, overwrite=True)
        #                 else:
        #                     if self.verbose > 0:
        #                         print('\nEpoch %05d: %s did not improve' %
        #                               (epoch + 1, self.monitor))
        #         else:
        #             if self.verbose > 0:
        #                 print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        #             if self.save_weights_only:
        #                 self.model.save_weights(filepath, overwrite=True)
        #             else:
        #                 self.model.save(filepath, overwrite=True)
        #
        #
        # def myFunc(self, current, filepath, epoch, mon, batch):
        #     import sys
        #     sys.path.insert(0, 'D:\\XinYu\\CNN\\RandomSampling')
        #     import TrainModel1
        #     if (self.TorV == 'Training' and round(current, 3) > TrainModel1.highestTrainingAcc):
        #         print("Training準確率高於歷史準確率-------------------------------儲存，第 " + str(mon) + " 個蒙地卡羅，第 " + str(epoch) + " 個Epoch")
        #         TrainModel1.highestTrainingAcc = round(current, 3)
        #         text_file = open('D:\\XinYu\\CNN\\RandomSampling\\HighestTraining.txt', 'w')
        #         text_file.write(str(TrainModel1.highestTrainingAcc) + "\n")
        #         text_file.write('蒙地卡羅第' + str(self.montecarlo) + '個隨機樣本集' + '\n')
        #         text_file.write(str(self.montecarlo))
        #         text_file.close()
        #         if self.save_weights_only:
        #             self.model.save_weights(filepath, overwrite=True)
        #         else:
        #             self.model.save(filepath, overwrite=True)
        #     elif (self.TorV == 'Validation' and round(current, 3) > TrainModel1.highestValidationAcc):
        #         print("Validation準確率高於歷史準確率-------------------------------儲存，第 " + str(mon) + " 個蒙地卡羅，第 " + str(epoch) + " 個Epoch")
        #         TrainModel1.highestValidationAcc = round(current, 3)
        #         text_file = open('D:\\XinYu\\CNN\\RandomSampling\\HighestValidation.txt', 'w')
        #         text_file.write(str(TrainModel1.highestValidationAcc) + "\n")
        #         text_file.write('蒙地卡羅第' + str(self.montecarlo) + '個隨機樣本集' + '\n')
        #         text_file.write(str(self.montecarlo))
        #         text_file.close()
        #         if self.save_weights_only:
        #             self.model.save_weights(filepath, overwrite=True)
        #         else:
        #             self.model.save(filepath, overwrite=True)

        #-----------------------------------------------------------------------------------------

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            callbacks=[historyPic,checkpointForHighestTraining,checkpointForHighestValidation]

        )



        #跑完一個蒙地卡羅 已存好最高準確率的epoch的model

        endTime = datetime.datetime.now()-startTime

        print("\n")
        print('training accuracy history: ', history.history['acc'])
        print('validation accuracy history: ', history.history['val_acc'])
        print("\n")
        sheet.write('A1', "TrainingData:" + str(imported_module.config_TrainingDataAmount) + "%,ValidationData:" + str(imported_module.config_ValidationDataAmount) + "%,TestingData:" + str(100 - imported_module.config_TrainingDataAmount - imported_module.config_ValidationDataAmount) + "%")
        sheet.write('B1', 'Training Accuracy')
        sheet.write('C1', 'Validation Accuracy')
        sheet.write('D1','Testing Accuracy')
        sheet.write('E1','Time Spending')
        sheet.write('A' + str(montecarlo + 2), '蒙地卡羅第' + str(montecarlo + 1) + '個隨機樣本集結果')
        sheet.write('B' + str(montecarlo + 2), round(max(history.history['acc']), 3))
        sheet.write('C' + str(montecarlo + 2), round(max(history.history['val_acc']), 3))
        sheet.write('E'+str(montecarlo+2),str(endTime).split(".")[0])



        if(len(TrainingEachMon)>=1 and len(ValidationEachMon)>=1):
            if(max(TrainingEachMon)<round(max(history.history['acc']), 3)):
                TMon = montecarlo+1
            if(max(ValidationEachMon)<round(max(history.history['val_acc']), 3)):
                VMon = montecarlo+1
            deleteFolder('D:\\Maimy\\CNN\\TestDataSet\\batchTime' + batchTime, TMon, VMon,batchTime)
            deleteFolder('D:\\Maimy\\CNN\\TrainingDataSet\\batchTime' + batchTime, TMon, VMon,batchTime)
            deleteFolder('D:\\Maimy\\CNN\\ValidationDataSet\\batchTime' + batchTime, TMon, VMon,batchTime)
        else:
            pass
        TrainingEachMon.append(round(max(history.history['acc']), 3))
        ValidationEachMon.append(round(max(history.history['val_acc']), 3))






    from keras.utils import plot_model

    plot_model(model, to_file='model.png')



    TrainingBestInExcel = ''
    ValidationBestInExcel=''
    TrainingBestInExcelAcc=''
    ValidationBestInExcelAcc=''


    fp = open("D:\\Maimy\\CNN\\RandomSampling\\HighestTraining.txt")
    print("Training 紀錄")
    for i, line in enumerate(fp):
        if i == 0:
         line = line.replace("\n","")
         TrainingBestInExcelAcc=line
         print("此次批次最高準確率 : "+line)
        if i == 1:
         line = line.replace("\n","")
         print("所屬蒙地卡羅 "+line)
        if i == 2:
         line = line.replace("\n","")
         TrainingBestInExcel = line
         TrainingTimes = line
    fp.close()

    fp = open("D:\\Maimy\\CNN\\RandomSampling\\HighestValidation.txt")
    print("Validation 紀錄")
    for i, line in enumerate(fp):
        if i == 0:
            line = line.replace("\n", "")
            ValidationBestInExcelAcc=line
            print("此次批次最高準確率 : " + line)
        if i == 1:
            line = line.replace("\n", "")
            print("所屬蒙地卡羅  " + line )
        if i == 2:
         line = line.replace("\n","")
         ValidationBestInExcel = line
         ValidationTimes = line
    fp.close()

    print("Training每次蒙地卡羅準確率 : ",TrainingEachMon)
    print()
    print("Validation妹次蒙地卡羅準確率 : ",ValidationEachMon)

    os.rename('D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File\\' + imported_module.config_Save_Model_File_Name + '_highestTrainingAcc.h5',
              'D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File\\' + imported_module.config_Save_Model_File_Name + '_highestTrainingAcc_' + today + '_MonTimes' + TrainingTimes + '_BatchTimes' + batchTime + '.h5')
    os.rename('D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File\\' + imported_module.config_Save_Model_File_Name + '_highestValidationAcc.h5',
              'D:\\Maimy\\CNN\\RandomSampling\\YourExcelFileAndH5File\\' + imported_module.config_Save_Model_File_Name + '_highestValidationAcc_' + today + '_MonTimes' + ValidationTimes + '_BatchTimes' + batchTime + '.h5')


    format = workbook.add_format()
    format2 = workbook.add_format()
    format2.set_font_size('20')
    format.set_bg_color('yellow')
    format.set_font_size('20')
    format.set_bold()
    sheet.write('B' + str(int(TrainingBestInExcel)+1),TrainingBestInExcelAcc, format)
    sheet.write('C' +str(int(ValidationBestInExcel)+1), ValidationBestInExcelAcc, format)
    sheet.set_column('A:A', 100,format2)
    sheet.set_column('B:B', 40, format2)
    sheet.set_column('C:C', 40, format2)
    sheet.set_column('D:D',40,format2)
    sheet.set_column('E:E', 60, format2)
    batchEndTime = datetime.datetime.now() - batchStartTime

    sheet.write('E' +str(int(imported_module.config_Monte_Carlo_time) + 2),str(batchEndTime).split(".")[0])
    runTesting(sheet,batchTime)

    workbook.close()



else:
    highestTrainingAcc = 0
    highestValidationAcc = 0
    TrainingTimes = ""
    ValidationTimes = ""













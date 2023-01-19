#================================================================================================
# Import important libraries
#================================================================================================
import paho.mqtt.client as mqtt
import time, queue, sys, datetime, json, math, scipy, pywt, time
import pandas as pd
import numpy as np

from json import JSONEncoder
from statistics import mode
from scipy import stats
from sklearn import preprocessing
from collections import defaultdict, Counter
from window_slider import *


from feature_extraction_utils import *
from data_reading_utils import *
from model_creation import *
from Numpy_to_JSON_utils import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import pandas as pd
import keras
import pickle


import pickle
import os
#import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images

import pandas as pd
import xlrd

#import tensorflow as tf
from keras.models import Model
from keras import layers
from keras import Input
from keras.utils.vis_utils import plot_model

#from tensorflow.keras import layers
from keras.layers import Conv1D, Conv2D, Conv3D, GRU, GlobalAveragePooling1D, Activation, Flatten, Dropout, Dense, MaxPool1D, MaxPool2D, Concatenate, ZeroPadding2D, ZeroPadding1D, Reshape
# import BatchNormalization
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend

from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations



from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout



from keras_preprocessing.image import img_to_array, load_img


from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers.recurrent import LSTM                                     check later 
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers.legacy import Adam
#from keras.layers.wrappers import TimeDistributed                            check later 
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation,Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from collections import deque
import time
import sys
#from tensorflow.keras import layers
from keras.layers import Conv1D, Conv2D, Conv3D, GRU, GlobalAveragePooling1D, Activation, Flatten, Dropout, Dense, MaxPool1D, MaxPool2D, Concatenate, ZeroPadding2D, ZeroPadding1D, Reshape
# import BatchNormalization
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.applications import resnet_v2

from keras.layers import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose, Flatten, Reshape, LeakyReLU
from keras.models import Model
from sklearn.model_selection import train_test_split
import random 
# from multi_label_performance_metrics_utils import *



#=================================================================================================
all_emg = []
all_eog = []
all_gsr = []
#=================================================================================================

#=================================================================================================
print('---------------------------------------------------------------')
n = sys.argv[1] #Reading the command line argument passed ['filename.py','passed value/client number']

client_name = 'LocalServer (User)'+n

print(client_name +':>>' +' ' +'Streaming Started!')

p = int(n) #Person number

#=================================================================================================

#=================================================================================================
# All MQTT ones Here
#=================================================================================================
qLS_emg = queue.Queue() #Queue to store the received message in on_message call back
qLS_eog = queue.Queue()
qLS_gsr = queue.Queue()


qLS_valence=queue.Queue()
qLS_arousal=queue.Queue()

def on_connect(client, userdata, flags, rc):
    if rc ==0:
        print("Local Server connected to broker successfylly ")
    else:
        print(f"Failed with code {rc}")

    for i in topic_list:
        val = client.subscribe(i)
        print(val)


def on_message(client, userdata, message): #On message callback from MQTT
    if (message.topic == "GlobalModel_EMG"):
        print("Message received from Global Model for EMG")
        qLS_emg.put(message)
    if (message.topic == "GlobalModel_EOG"):
        print("Message received from Global Model for EOG")
        qLS_eog.put(message)
    if (message.topic == "GlobalModel_GSR"):
        print("Message received from Global Model for GSR")
        qLS_gsr.put(message)
    if (message.topic == "GlobalModel_Valence"):
        print("Message received from Valence Classification")
        qLS_valence.put(message)
    if (message.topic == "GlobalModel_Arousal"):
        print("Message received from Arousal Classification")
        qLS_arousal.put(message)


# mqttBroker = "mqtt.eclipseprojects.io" #Used MQTT Broker
mqttBroker = "127.0.0.1"

client = mqtt.Client(client_name) #mqtt Client
client.on_connect = on_connect
client.connect(mqttBroker, 1883) #mqtt broker connect

topic_list =[('GlobalModel_EMG',0),('GlobalModel_EOG',0),('GlobalModel_GSR',0),('GlobalModel_Valence',0),('GlobalModel_Arousal',0)] #Subscription topic list

client.loop_start()

#**********************************************************
time.sleep(5) #Wait for connection setup to complete
#**********************************************************

print('---------------------------------------------------------------')

#=================================================================================================

#------------------------------------
# Once file fetched data stored here
#------------------------------------
# grand_eeg = eeg_data(p)

EMG_all,EOG_all,GSR_all,label_data_all = get_emg_eog_gsr_labels_data(p)
# grand_eda = eda_data(p)
# grand_resp = resp_data(p)

#=================================================================================================


#=================================================================================================
# Sliding Window
#=================================================================================================

segment_in_sec = 20 #in sec
bucket_size = int((8064/60)*segment_in_sec)  #8064 is for 60 sec record
overlap_count = 0

#================================================
# Model name and other loop control parameters
#================================================
classifier = 'FFNN_Feature_Fusion'
init_m = 0
indx = 0
c=0
ccc =0
i =0
videos = 40 #Total Number of Videos
#=================================================================================================

print('-----------------------------------------')
print('Working with -->', classifier)
print('-----------------------------------------')
#=======================================
# MAIN Loop STARTS HERE
#=======================================

#vid=[]
#for m in range(40):
#	vid.append(m)

#vid_train=random.sample(vid, 30)

vid_train=2
#vid_test=10
vid_test_last=4
for jj in range(vid_train): #Video loop for each participants Replce 6 with vidoes if you want all 40 
    v = jj+1 #Video number
    print('=========================================================================')
    p_v = 'Person:'+ ' ' +str(p)+ ' ' +'Video:'+str(v)
    print(p_v)

    emotion_label =[]
    
    t_EMG,t_EOG,t_GSR,y = get_data_video(jj,EMG_all,EOG_all,GSR_all,label_data_all)
    x_train_emg, x_test_emg, y_train_emg, y_test_emg = train_test_split(t_EMG, t_EMG, test_size=0.2, random_state=42)
    x_train_eog, x_test_eog, y_train_eog, y_test_eog = train_test_split(t_EOG, t_EOG, test_size=0.2, random_state=42)
    x_train_gsr, x_test_gsr, y_train_gsr, y_test_gsr = train_test_split(t_GSR, t_GSR, test_size=0.2, random_state=42)

        #===================================================
        # Model initialization
        #===================================================
    if init_m == 0:
        fm_model_emg = emg_model()
        fm_model_eog = eog_model()
        fm_model_gsr = gsr_model()
        

        init_m = init_m+1


        #===============================================================
        # Emotion Classification --> Valence and Arousal
        #===============================================================

#     if c == 0: #For the first time model will return 9 or None
#         tmp_y = np.array([[9,9]])
        
        
        
# #         fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

#         c=c+1

#     else:
#         tmp_y = fm_model.predict(x_FF)
#         fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

    fm_model_emg.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
    early_stopper = EarlyStopping(patience=10, restore_best_weights=True)

    start_time=time.time()
    history1 = fm_model_emg.fit(x_train_emg, x_train_emg,
                        epochs=3,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(x_test_emg, x_test_emg),
                        callbacks=[early_stopper])
                        
    model1= 's_all_reconstructed_emg_encoded'+str(p)+'.h5'
    #encoder_emg = Model(input0, encoded_emg)
    OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"

    try:
    	os.remove(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    except:
    	pass
    fm_model_emg.save(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    #model_weights_emg = fm_model_emg.get_weights()
    #encodedModelWeights_emg = json.dumps(model_weights_emg,cls=Numpy2JSONEncoder)
    client.publish("LocalModel_for_EMG_Signal saved", payload = 'abc')
    #save model
    
    fm_model_eog.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
    early_stopper = EarlyStopping(patience=10, restore_best_weights=True)

    history1 = fm_model_eog.fit(x_train_eog, x_train_eog,
                        epochs=3,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(x_test_eog, x_test_eog),
                        callbacks=[early_stopper])
    
    model1= 's_all_reconstructed_eog_encoded'+str(p)+'.h5'
    #encoder_emg = Model(input0, encoded_emg)
    OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
    

    try:
    	os.remove(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    except:
    	pass
    fm_model_eog.save(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    #model_weights_emg = fm_model_emg.get_weights()
    #encodedModelWeights_emg = json.dumps(model_weights_emg,cls=Numpy2JSONEncoder)
    client.publish("LocalModel_for_EOG_Signal saved", payload = 'abc')
    
    
    fm_model_gsr.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
    early_stopper = EarlyStopping(patience=10, restore_best_weights=True)

    history1 = fm_model_gsr.fit(x_train_gsr, x_train_gsr,
                        epochs=3,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(x_test_gsr, x_test_gsr),
                        callbacks=[early_stopper])
                        
#     tmp_y = fm_model.predict(x_train_emg)

#         if slider_eda.reached_end_of_list():
#             break


    #===========================================
    # Performance matric update
    #===========================================
    
    model1= 's_all_reconstructed_gsr_encoded'+str(p)+'.h5'
    #encoder_emg = Model(input0, encoded_emg)
    OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
    

    try:
    	os.remove(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    except:
    	pass
    fm_model_gsr.save(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    #model_weights_emg = fm_model_emg.get_weights()
    #encodedModelWeights_emg = json.dumps(model_weights_emg,cls=Numpy2JSONEncoder)
    client.publish("LocalModel_for_GSR_Signal saved", payload = 'abc')
    
    end_time=time.time()
    print('time taken for local training autoencoders ',end_time-start_time)
    
    print('All models saved')
    start_time=time.time()
    
    results_emg = fm_model_emg.evaluate(x_test_emg, y_test_emg)
    print("Test loss for EMG signals: ", results_emg)
    results_eog = fm_model_eog.evaluate(x_test_eog, y_test_eog)
    print("Test loss for EOG signals: ", results_eog)
    results_gsr = fm_model_gsr.evaluate(x_test_gsr, y_test_gsr)
    print("Test loss for GSR signals: ", results_gsr)
    
#     y_pred = np.array([np.argmax(tmp_y[0])])

#     mc_y_act = np.array([np.argmax(y_act)])

#     bac = accuracy_score(mc_y_act,y_pred)
#     f1 = f1_score(mc_y_act,y_pred, average='micro')

#     print('-------------------------------------------------------------------------------')
#     print('Actual Class:',discrete_emotion[mc_y_act[0]])
#     print('Fusion Model predicted:{}'.format(discrete_emotion[y_pred[0]]))


#     print(client_name+'-->'+'Accuracy:{}'.format(bac))
#     print(client_name+'-->'+'F1-score:{}'.format(f1))
    print('-------------------------------------------------------------------------------')

#     all_emo.append([p,v, bac,f1, mc_y_act[0], y_pred[0]])
    all_emg.append(results_emg)
    all_eog.append(results_eog)
    all_gsr.append(results_gsr)

    #=======================================================================================
    #Send the model performance from each to server for checking Global Model's performance
    #========================================================================================
    if i >=0:
        model_performance = {'Local_Model':p,'Loss_EMG':results_emg, 'Loss_EOG':results_eog,'Loss_GSR':results_gsr}
        encoded_model_performance = json.dumps(model_performance)
        client.publish("ModelPerformance", payload = encoded_model_performance)
        print("Local Model Performance Broadcasted for "+ p_v +" to Topic:-> ModelPerformance")



    #==========================================================
    # Model weight compress into JSON format
    #==========================================================

    #Message Generation and Encoding into JSON
    
    
    



    #==========================================================
    # Broadcast (Publish) Local model weights to the mqttBroker
    #==========================================================

    
    
    

    print("Local Model Broadcasted for "+ p_v +" to Topic:-> LocalModel")
    
    end_time=time.time()
    print('time taken for testing,sending is ',end_time-start_time)
    print('Waiting for global models')
    
    client.on_message = on_message   #added here

    #**********************************************************
    time.sleep(100) #put the loca server in sleep for 60 sec  #82
    #**********************************************************

    #===============================================================================
    # Receive Global model from the Subscriber end
    #===============================================================================

    # if i>0:
    #===============================================================================
    # Publisher as subscriber to receive results after operation at Subscriber end
    #===============================================================================
    #client.on_message = on_message   #changed
    while not qLS_emg.empty():
        message = qLS_emg.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')

        # Deserialization the encoded received JSON data
        model1='s_all_reconstructed_emg_encoded_global.h5'
        fm_model_emg.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
        print('loaded global weights emg') 
        #Replacing the old model with the newley received model from Global Server

#         fm_model_eog.set_weights(global_weights)
#         fm_model_gsr.set_weights(global_weights)

    while not qLS_eog.empty():
        message = qLS_eog.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')

        # Deserialization the encoded received JSON data
        model1='s_all_reconstructed_eog_encoded_global.h5'
        fm_model_eog.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
        print('loaded global weights eog')
        
    while not qLS_gsr.empty():
        message = qLS_gsr.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')

        # Deserialization the encoded received JSON data
        model1='s_all_reconstructed_gsr_encoded_global.h5'
        fm_model_gsr.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
        print('loaded global weights gsr')
    
    
    
    if (i == videos): #if all the videos are done means no more data from User
        break

    i +=1

# model1 = 's_all_reconstructed_emg_autoencoder_1.h5'
# fm_model_emg.save(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))

# model2 = 's_all_reconstructed_eog_autoencoder_1.h5'
# fm_model_eog.save(os.path.join(OUTPUT, 'Models', 'autoencoders', model2))
# model3 = 's_all_reconstructed_gsr_autoencoder_1.h5'
# fm_model_gsr.save(os.path.join(OUTPUT, 'Models', 'autoencoders', model3))


    
#===============================================================================
#Save all the results into CSV file
#===============================================================================
# folderPath = '/home/bits/Downloads/Fed-ReMECS-mqtt/Federated_Results/'
# fname_fm = folderPath + client_name +'_person_FusionModel'+'_'+'_results.csv'
# column_names = ['Person', 'Video', 'Loss_','F1', 'y_act', 'y_pred']
# all_emo = pd.DataFrame(all_emo,columns = column_names)
# all_emo.to_csv(fname_fm)

print('loss are')
print('EMG: ',all_emg,'EOG: ',all_eog,'GSR: ',all_gsr)
print('All Done! Client Closed')


#------------TESTING----------
x_test_emg_all=None
x_test_eog_all=None
x_test_gsr_all=None
model1='s_all_reconstructed_emg_encoded_global.h5'
fm_model_emg.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
model1='s_all_reconstructed_eog_encoded_global.h5'
fm_model_eog.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
model1='s_all_reconstructed_gsr_encoded_global.h5'
fm_model_gsr.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
for jj in range(vid_train,vid_test_last): 
    
    t_EMG,t_EOG,t_GSR,y = get_data_video(jj,EMG_all,EOG_all,GSR_all,label_data_all)
    
    
    if x_test_emg_all is None:
    	x_test_emg_all=t_EMG
    	x_test_eog_all=t_EOG
    	x_test_gsr_all=t_GSR
    	
    else:
    	x_test_emg_all=np.concatenate((x_test_emg_all,t_EMG),axis=0)
    	x_test_eog_all=np.concatenate((x_test_eog_all,t_EOG),axis=0)
    	x_test_gsr_all=np.concatenate((x_test_gsr_all,t_GSR),axis=0)
    	
results_emg = fm_model_emg.evaluate(x_test_emg_all, x_test_emg_all)
print("Test loss for EMG signals: ", results_emg)
results_eog = fm_model_eog.evaluate(x_test_eog_all, x_test_eog_all)
print("Test loss for EOG signals: ", results_eog)
results_gsr = fm_model_gsr.evaluate(x_test_gsr_all, x_test_gsr_all)
print("Test loss for GSR signals: ", results_gsr)

#Global Model Result Save
from csv import writer
 
# List that we want to add as a new row
List = [p,results_emg,results_eog,results_gsr]
 

with open('/home/csis/Documents/Fed-ReMECS-mqtt-main/Federated_Results/results.csv', 'a') as f_object:
 
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)
 
    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(List)
 
    # Close the file object
    f_object.close()


print('---------------Start valence----------------')

i_vid=0
init_m=0
fm_model_emg = emg_model()
fm_model_eog = eog_model()
fm_model_gsr = gsr_model()
OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
model1='s_all_reconstructed_emg_encoded_global.h5'
fm_model_emg.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
global_weight_emg=fm_model_emg.get_weights()

model1='s_all_reconstructed_eog_encoded_global.h5'
fm_model_eog.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
global_weight_eog=fm_model_eog.get_weights()

model1='s_all_reconstructed_gsr_encoded_global.h5'
fm_model_gsr.load_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
global_weight_gsr=fm_model_gsr.get_weights()

#===============================================================================
#For valence 
#===============================================================================
for jj in range(0,vid_train): #Video loop for each participants
    v = jj+1 #Video number
    print('=========================================================================')
    p_v = 'Person:'+ ' ' +str(p)+ ' ' +'Video:'+str(v)
    print(p_v)

    emotion_label =[]
    
    t_EMG,t_EOG,t_GSR,y = get_data_video(jj,EMG_all,EOG_all,GSR_all,label_data_all)

    from sklearn.model_selection import train_test_split

    X2_train, X2_test, y2_train, y2_test = train_test_split(t_EMG, y[:,0], test_size=0.2, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(t_EOG, y[:,0], test_size=0.2, random_state=42)
    X4_train, X4_test, y4_train, y4_test = train_test_split(t_GSR, y[:,0], test_size=0.2, random_state=42)
    
    val = []
    val.append(X3_train)
    val.append(X2_train)
    val.append(X4_train)

    X_train = val

    val = []
    val.append(X3_test)
    val.append(X2_test)
    val.append(X4_test)
    X_test = val

    y_train = y2_train
    y_test = y2_test

        #===================================================
        # Model initialization
        #===================================================
    if init_m == 0:
        model=valence_classification(global_weight_emg,global_weight_eog,global_weight_gsr)
        init_m = init_m+1


        #===============================================================
        # Emotion Classification --> Valence and Arousal
        #===============================================================

#     if c == 0: #For the first time model will return 9 or None
#         tmp_y = np.array([[9,9]])
        
        
        
# #         fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

#         c=c+1

#     else:
#         tmp_y = fm_model.predict(x_FF)
#         fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

#     fm_model_emg.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
    batch_size = 32
    nb_epoch = 10	
    early_stopper = EarlyStopping(patience=20, restore_best_weights=True)
    history10 = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                # callbacks=[early_stopper],
                epochs=nb_epoch)
    
    
   

    from sklearn.metrics import classification_report, confusion_matrix

    y_true = y_test

    y_pred = model.predict(X_test)
    t = []
    for i in y_pred:
        if i[0]>=i[1]:
            t.append([1,0])
        else:
            t.append([0,1])

    target_names = ['Low Valence','High Valence']
    c1 = classification_report(y_true, t, target_names=target_names,output_dict=True)
    print(c1)
    
#     tmp_y = fm_model.predict(x_train_emg)

#         if slider_eda.reached_end_of_list():
#             break


    #===========================================
    # Performance matric update
    #===========================================
    
#     results_emg = fm_model.evaluate(x_test_emg, y_test_emg)
#     print("Test loss for EMG signals: ", results_emg)
#     results_eog = fm_model.evaluate(x_test_eog, y_test_eog)
#     print("Test loss for EOG signals: ", results_eog)
#     results_gsr = fm_model.evaluate(x_test_gsr, y_test_gsr)
#     print("Test loss for GSR signals: ", results_gsr)
    
#     y_pred = np.array([np.argmax(tmp_y[0])])

#     mc_y_act = np.array([np.argmax(y_act)])

#     bac = accuracy_score(mc_y_act,y_pred)
#     f1 = f1_score(mc_y_act,y_pred, average='micro')

#     print('-------------------------------------------------------------------------------')
#     print('Actual Class:',discrete_emotion[mc_y_act[0]])
#     print('Fusion Model predicted:{}'.format(discrete_emotion[y_pred[0]]))


#     print(client_name+'-->'+'Accuracy:{}'.format(bac))
#     print(client_name+'-->'+'F1-score:{}'.format(f1))
    print('-------------------------------------------------------------------------------')

#     all_emo.append([p,v, bac,f1, mc_y_act[0], y_pred[0]])
#     all_emg.append(results_emg)
#     all_eog.append(results_eog)
#     all_gsr.append(results_gsr)

    #=======================================================================================
    #Send the model performance from each to server for checking Global Model's performance
    #========================================================================================
    if i_vid >=0:
        model_performance = {'Local_Model':p,'Low_Valence_precision':c1['Low Valence']['precision'],'High_Valence_precision':c1['High Valence']['precision'],'Low_Valence_recall':c1['Low Valence']['recall'],'High_Valence_recall':c1['High Valence']['recall'],'Low_Valence_f1-score':c1['Low Valence']['f1-score'], 'High_Valence_f1-score':c1['High Valence']['f1-score']}
        encoded_model_performance = json.dumps(model_performance)
        client.publish("ModelPerformance_valence", payload = encoded_model_performance)
        print("Local Model Performance Broadcasted for "+ p_v +" to Topic:-> ModelPerformance_valence")



    #==========================================================
    # Model weight compress into JSON format
    #==========================================================

    #Message Generation and Encoding into JSON
    #model_weights_valence = model.get_weights()
    #encodedModelWeights_valence = json.dumps(model_weights_valence,cls=Numpy2JSONEncoder)

    #==========================================================
    # Broadcast (Publish) Local model weights to the mqttBroker
    #==========================================================
    
    
    
    
    
    
    model1= 's_all_reconstructed_valence_encoded'+str(p)+'.h5'
    #encoder_emg = Model(input0, encoded_emg)
    OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"

    try:
    	os.remove(os.path.join(OUTPUT, 'Models', 'valence', model1))
    except:
    	pass
    model.save(os.path.join(OUTPUT, 'Models', 'valence', model1))
    #model_weights_emg = fm_model_emg.get_weights()
    #encodedModelWeights_emg = json.dumps(model_weights_emg,cls=Numpy2JSONEncoder)
    
    client.publish("LocalModel_for_Valence", payload = 'abc')
    

    print("Local Model Broadcasted for "+ p_v +" to Topic:-> LocalModel")

    #**********************************************************
    print('waiting for global model')
    time.sleep(20) #put the loca server in sleep for 60 sec
    client.on_message = on_message
    while not qLS_valence.empty():
    	message = qLS_valence.get()
    	if message is None:
    		continue
    	msg = message.payload.decode('utf-8')
    	model1='s_all_reconstructed_valence_encoded_global.h5'
    	model.load_weights(os.path.join(OUTPUT, 'Models', 'valence', model1))
    	print('got global weights')
    if(i_vid == videos):
    	break
    i_vid +=1


#------------TESTING----------
x_test_emg_all=None
x_test_eog_all=None
x_test_gsr_all=None
y_all=None

model1='s_all_reconstructed_valence_encoded_global.h5'
model.load_weights(os.path.join(OUTPUT, 'Models', 'valence', model1))

for jj in range(vid_train,vid_test_last): 
    
    t_EMG,t_EOG,t_GSR,y = get_data_video(jj,EMG_all,EOG_all,GSR_all,label_data_all)
    
    
    if x_test_emg_all is None:
    	x_test_emg_all=t_EMG
    	x_test_eog_all=t_EOG
    	x_test_gsr_all=t_GSR
    	y_all=y[:,0]
    	
    else:
    	x_test_emg_all=np.concatenate((x_test_emg_all,t_EMG),axis=0)
    	x_test_eog_all=np.concatenate((x_test_eog_all,t_EOG),axis=0)
    	x_test_gsr_all=np.concatenate((x_test_gsr_all,t_GSR),axis=0)
    	y_all=np.concatenate((y_all,y[:,0]),axis=0)
    	
test = []
test.append(x_test_eog_all)
test.append(x_test_emg_all)
test.append(x_test_gsr_all)

X_test = test

from sklearn.metrics import classification_report, confusion_matrix

y_true = y_all

y_pred = model.predict(X_test)
t = []
for i in y_pred:
	if i[0]>=i[1]:
		t.append([1,0])
	else:
		t.append([0,1])

target_names = ['Low Valence','High Valence']
c1 = classification_report(y_true, t, target_names=target_names,output_dict=True)
print(c1)

#Global Model Result Save
from csv import writer
 
# List that we want to add as a new row
List =[p,c1['Low Valence']['precision'],c1['High Valence']['precision'],c1['Low Valence']['recall'],c1['High Valence']['recall'],c1['Low Valence']['f1-score'], c1['High Valence']['f1-score']]
 

with open('/home/csis/Documents/Fed-ReMECS-mqtt-main/Federated_Results/valence_results.csv', 'a') as f_object:
 
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)
 
    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(List)
 
    # Close the file object
    f_object.close()


#===============================================================================
#For arousal classification
#===============================================================================

print('----------------------STARTING AROUSAL TRAINING---------------')

i_vid=0
for jj in range(0,vid_train): #Video loop for each participants
    v = jj+1 #Video number
    print('=========================================================================')
    p_v = 'Person:'+ ' ' +str(p)+ ' ' +'Video:'+str(v)
    print(p_v)

    emotion_label =[]
    
    t_EMG,t_EOG,t_GSR,y = get_data_video(jj,EMG_all,EOG_all,GSR_all,label_data_all)

    from sklearn.model_selection import train_test_split
    X2_train_a, X2_test_a, y2_train_a, y2_test_a = train_test_split(t_EMG, y[:,1], test_size=0.2, random_state=42)
    X3_train_a, X3_test_a, y3_train_a, y3_test_a = train_test_split(t_EOG, y[:,1], test_size=0.2, random_state=42)
    X4_train_a, X4_test_a, y4_train_a, y4_test_a = train_test_split(t_GSR, y[:,1], test_size=0.2, random_state=42)
    
    val = []
    val.append(X3_train)
    val.append(X2_train)
    val.append(X4_train)

    X_train = val

    val = []
    val.append(X3_test)
    val.append(X2_test)
    val.append(X4_test)
    X_test = val

    y_train = y2_train
    y_test = y2_test

        #===================================================
        # Model initialization
        #===================================================
    if init_m == 0:
        model=arousal_model(global_weight_emg,global_weight_eog,global_weight_gsr)
        init_m = init_m+1


        #===============================================================
        # Emotion Classification --> Valence and Arousal
        #===============================================================

#     if c == 0: #For the first time model will return 9 or None
#         tmp_y = np.array([[9,9]])
        
        
        
# #         fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

#         c=c+1

#     else:
#         tmp_y = fm_model.predict(x_FF)
#         fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

#     fm_model_emg.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
    batch_size = 32
    nb_epoch = 10	
    early_stopper = EarlyStopping(patience=20, restore_best_weights=True)
    history10 = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                # callbacks=[early_stopper],
                epochs=nb_epoch)
    
    
   

    from sklearn.metrics import classification_report, confusion_matrix
    y_true = y_test
    y_pred = model.predict(X_test)
    t = []
    for i in y_pred:
    	if i[0]>=i[1]:
    		t.append([1,0])
    	else:
    		t.append([0,1])
    target_names = ['Low Arousal','High Arousal']
    c2 = classification_report(y_true, t, target_names=target_names,output_dict=True)
    print(c2)
    
#     tmp_y = fm_model.predict(x_train_emg)

#         if slider_eda.reached_end_of_list():
#             break


    #===========================================
    # Performance matric update
    #===========================================
    
#     results_emg = fm_model.evaluate(x_test_emg, y_test_emg)
#     print("Test loss for EMG signals: ", results_emg)
#     results_eog = fm_model.evaluate(x_test_eog, y_test_eog)
#     print("Test loss for EOG signals: ", results_eog)
#     results_gsr = fm_model.evaluate(x_test_gsr, y_test_gsr)
#     print("Test loss for GSR signals: ", results_gsr)
    
#     y_pred = np.array([np.argmax(tmp_y[0])])

#     mc_y_act = np.array([np.argmax(y_act)])

#     bac = accuracy_score(mc_y_act,y_pred)
#     f1 = f1_score(mc_y_act,y_pred, average='micro')

#     print('-------------------------------------------------------------------------------')
#     print('Actual Class:',discrete_emotion[mc_y_act[0]])
#     print('Fusion Model predicted:{}'.format(discrete_emotion[y_pred[0]]))


#     print(client_name+'-->'+'Accuracy:{}'.format(bac))
#     print(client_name+'-->'+'F1-score:{}'.format(f1))
    print('-------------------------------------------------------------------------------')

#     all_emo.append([p,v, bac,f1, mc_y_act[0], y_pred[0]])
#     all_emg.append(results_emg)
#     all_eog.append(results_eog)
#     all_gsr.append(results_gsr)

    #=======================================================================================
    #Send the model performance from each to server for checking Global Model's performance
    #========================================================================================
    if i_vid >=0:
        model_performance = {'Local_Model':p,'Low_Arousal_precision':c2['Low Arousal']['precision'],'High_Arousal_precision':c2['High Arousal']['precision'],'Low_Arousal_recall':c2['Low Arousal']['recall'],'High_Arousal_recall':c2['High Arousal']['recall'],'Low_Arousal_f1-score':c2['Low Arousal']['f1-score'], 'High_Arousal_f1-score':c2['High Arousal']['f1-score']}
        encoded_model_performance = json.dumps(model_performance)
        client.publish("ModelPerformance_arousal", payload = encoded_model_performance)
        print("Local Model Performance Broadcasted for "+ p_v +" to Topic:-> ModelPerformance_Arousal")



    #==========================================================
    # Model weight compress into JSON format
    #==========================================================

    #Message Generation and Encoding into JSON
    #model_weights_valence = model.get_weights()
    #encodedModelWeights_valence = json.dumps(model_weights_valence,cls=Numpy2JSONEncoder)

    #==========================================================
    # Broadcast (Publish) Local model weights to the mqttBroker
    #==========================================================
    
    
    
    
    
    
    model2= 's_all_reconstructed_arousal_encoded'+str(p)+'.h5'
    #encoder_emg = Model(input0, encoded_emg)
    OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"

    try:
    	os.remove(os.path.join(OUTPUT, 'Models', 'arousal', model2))
    except:
    	pass
    model.save(os.path.join(OUTPUT, 'Models', 'arousal', model2))
    #model_weights_emg = fm_model_emg.get_weights()
    #encodedModelWeights_emg = json.dumps(model_weights_emg,cls=Numpy2JSONEncoder)
    
    client.publish("LocalModel_for_Arousal", payload = 'abc')
    

    print("Local Model Broadcasted for "+ p_v +" to Topic:-> LocalModel")

    #**********************************************************
    print('waiting for global model')
    time.sleep(20) #put the loca server in sleep for 60 sec
    client.on_message = on_message
    while not qLS_arousal.empty():
    	message = qLS_arousal.get()
    	if message is None:
    		continue
    	msg = message.payload.decode('utf-8')
    	model1='s_all_reconstructed_arousal_encoded_global.h5'
    	model.load_weights(os.path.join(OUTPUT, 'Models', 'arousal', model1))
    	print('got global weights')
    if(i_vid == videos):
    	break
    i_vid +=1

#------------TESTING----------
x_test_emg_all=None
x_test_eog_all=None
x_test_gsr_all=None
y_all=None

model1='s_all_reconstructed_arousal_encoded_global.h5'
model.load_weights(os.path.join(OUTPUT, 'Models', 'arousal', model1))

for jj in range(vid_train,vid_test_last): 
    
    t_EMG,t_EOG,t_GSR,y = get_data_video(jj,EMG_all,EOG_all,GSR_all,label_data_all)
    
    
    if x_test_emg_all is None:
    	x_test_emg_all=t_EMG
    	x_test_eog_all=t_EOG
    	x_test_gsr_all=t_GSR
    	y_all=y[:,1]
    	
    else:
    	x_test_emg_all=np.concatenate((x_test_emg_all,t_EMG),axis=0)
    	x_test_eog_all=np.concatenate((x_test_eog_all,t_EOG),axis=0)
    	x_test_gsr_all=np.concatenate((x_test_gsr_all,t_GSR),axis=0)
    	y_all=np.concatenate((y_all,y[:,1]),axis=0)
    	
test = []
test.append(x_test_eog_all)
test.append(x_test_emg_all)
test.append(x_test_gsr_all)

X_test = test

from sklearn.metrics import classification_report, confusion_matrix
y_true = y_all
y_pred = model.predict(X_test)
t = []
for i in y_pred:
	if i[0]>=i[1]:
		t.append([1,0])
	else:
		t.append([0,1])
target_names = ['Low Arousal','High Arousal']
c2 = classification_report(y_true, t, target_names=target_names,output_dict=True)
print(c2)

#Global Model Result Save
from csv import writer
 
# List that we want to add as a new row
List =[p,c2['Low Arousal']['precision'],c2['High Arousal']['precision'],c2['Low Arousal']['recall'],c2['High Arousal']['recall'],c2['Low Arousal']['f1-score'],c2['High Arousal']['f1-score']]
 

with open('/home/csis/Documents/Fed-ReMECS-mqtt-main/Federated_Results/arousal_results.csv', 'a') as f_object:
 
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)
 
    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(List)
 
    # Close the file object
    f_object.close()    

print('All done')

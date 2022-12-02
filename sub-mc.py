import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
import time, queue, sys
import numpy as np
import pandas as pd
import json
from json import JSONEncoder
from federated_utils import *
from Numpy_to_JSON_utils import *
from feature_extraction_utils import *
from data_reading_utils import *
from model_creation import *

from keras.models import load_model

global qGSModel, qGSPerfm
qGSModel_emg = queue.Queue()
qGSModel_eog = queue.Queue()
qGSModel_gsr = queue.Queue()
qGSPerfm = queue.Queue()
qGSValence=queue.Queue()
qGSPerformance_Valence=queue.Queue()
qGSArousal=queue.Queue()
qGSPerformance_Arousal=queue.Queue()




global global_model_result_emg, prev_global_model_emg, current_global_model_emg,global_model_result_eog, prev_global_model_eog, current_global_model_eog,global_model_result_gsr, prev_global_model_gsr, current_global_model_gsr
global_model_result_emg =[]
prev_global_model_emg = list()
global_model_result_eog =[]
prev_global_model_eog = list()
global_model_result_gsr =[]
prev_global_model_gsr = list()

l_rate = 0.05 #Learning rate

n = int(sys.argv[1])
def on_connect(client, userdata, flags, rc):
    if rc ==0:
        print("Global Server connected to broker successfylly ")
    else:
        print(f"Failed with code {rc}")

    for i in topic_list:
        val = client.subscribe(i)
        print(val)


def on_message(client, userdata, message):
    if (message.topic == "LocalModel_for_EMG_Signal saved"):
        print("Message received from Local Model for EMG")
        qGSModel_emg.put(message)
    if (message.topic == "LocalModel_for_EOG_Signal saved"):
        print("Message received from Local Model for EOG")
        qGSModel_eog.put(message)
    if (message.topic == "LocalModel_for_GSR_Signal saved"):
        print("Message received from Local Model for GSR")
        qGSModel_gsr.put(message)

    if(message.topic == 'ModelPerformance'):
        print('Performance metric received  from Local Models')
        qGSPerfm.put(message)
        
    if(message.topic == 'ModelPerformance_valence'):
        print('Performance metric received  from Local Models')
        qGSPerformance_Valence.put(message)
    if (message.topic == "LocalModel_for_Valence"):
        print("Message received from Local Model for Valence")
        qGSValence.put(message)
    if(message.topic == 'ModelPerformance_arousal'):
        print('Performance metric received  from Local Models')
        qGSPerformance_Arousal.put(message)
    if (message.topic == "LocalModel_for_Arousal"):
        print("Message received from Local Model for Arousal")
        qGSArousal.put(message)  
       
    


print('---------------------------------------------------------------')
# mqttBroker = "mqtt.eclipseprojects.io"
mqttBroker = "127.0.0.1"
client = mqtt.Client(client_id ="GlobalServer", clean_session=True)
client.on_connect = on_connect
client.connect(mqttBroker,1883)

topic_list =[('LocalModel_for_EMG_Signal saved',0),('LocalModel_for_EOG_Signal saved',0),('LocalModel_for_GSR_Signal saved',0),('ModelPerformance',0),('LocalModel_for_Valence',0),('ModelPerformance_valence',0),('LocalModel_for_Arousal',0),('ModelPerformance_arousal',0)]

client.loop_start()
client.on_message = on_message

#**********************************************************
time.sleep(5) #Wait for connection setup to complete
#**********************************************************

print('---------------------------------------------------------------')


i = 0


while True:
    print('---------STARTED-------------')
    print('Global Server')
    # print('Round: ',i)

    
    #====================================================
    # Global Model Performance Printing
    #====================================================

    if (i>0): #after first round of model exchange global models performance is calculated
        print('waiting for performance metrics')
        time.sleep(10)
        print('Now Collecting Local Model performance Metrics....')
        local_model_performace = list()
        global_model_result=list()
        while not qGSPerfm.empty():
            message = qGSPerfm.get()

            if message is None:
                continue

            msg_model_performance = message.payload.decode('utf-8')

            decodedModelPerfromance = list(json.loads(msg_model_performance).values())
            local_model_performace.append(decodedModelPerfromance)

        global_model_performance = np.array(local_model_performace)
        global_performance = np.mean(global_model_performance, axis=0)

        len_local_perfm = len(local_model_performace)
        print('Total Model Performance received:',len_local_perfm)

        if (len_local_perfm != 0):
            global_model_result.append([global_performance[1],global_performance[2],global_performance[3]])
            print('----------------------------------------------------')
            print('Loss_EMG:',global_performance[1])
            print('Loss_EOG:',global_performance[2])
            print('Loss_GSR:',global_performance[3])
            print('----------------------------------------------------')
        else:
            break #No more data from local model

    #**********************************************************
    time.sleep(8) #to receive model weights
    #**********************************************************

    #=========================================================
    # Local Model Receiving Part
    #=========================================================
    all_local_model_weights_emg = list()
    model_available=0
    
    while not qGSModel_emg.empty():
            message = qGSModel_emg.get()

            if message is None:
                continue

            msg_model = message.payload.decode('utf-8')
            print('msg_model is ',msg_model)
            if(msg_model=='abc'):
            	model_available=1

    if(model_available==0):
    	print("There is no local model to receoive")
    	break
    
    for p in range(n):
    	model1= 's_all_reconstructed_emg_encoded'+str(p+1)+'.h5'
    	OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
    	encoder_emg = load_model(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    	local_model_weights_emg=encoder_emg.get_weights()
    	scaled_weights = scale_model_weights(local_model_weights_emg, 0.1)
    	all_local_model_weights_emg.append(scaled_weights)

    print('Total Local Model Received for EMG:',len(all_local_model_weights_emg))

    #======================================================

    i +=1 #Next round increment

    #===================================================================
    # Publish the Global Model after Federated Averaging
    #===================================================================
    if i >0:
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights_emg)

        fm_model_emg_global = emg_model() #temp model
        fm_model_emg_global.set_weights(averaged_weights)
        
        model1= 's_all_reconstructed_emg_encoded_global.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        try:
        	os.remove(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
        except:
        	pass
        fm_model_emg_global.save_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1),overwrite=True)

        client.publish("GlobalModel_EMG", payload = 'abc') #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model EMG to Topic:--> GlobalModel")

        #**********************************************************
        time.sleep(5)#pause it so that the publisher gets the Global model
        #**********************************************************

        #====================================================================


    print('---------------HERE------------------')
    #===================================================================================
    # If No more data from Publisher exit and server closed connection to the broker
    #===================================================================================
#     if(i >0 and len(all_local_model_weights_emg)==0): #loop break no message from producer
#         break
        
        
    time.sleep(8) #to receive model weights
    all_local_model_weights_eog = list()

    for p in range(n):
    	model1= 's_all_reconstructed_eog_encoded'+str(p+1)+'.h5'
    	OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
    	encoder_eog = load_model(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    	local_model_weights_eog=encoder_eog.get_weights()
    	scaled_weights = scale_model_weights(local_model_weights_eog, 0.1)
    	all_local_model_weights_eog.append(scaled_weights)

    print('Total Local Model Received for eog:',len(all_local_model_weights_eog))
        

    #======================================================

    i +=1 #Next round increment

    #===================================================================
    # Publish the Global Model after Federated Averaging
    #===================================================================
    if i >0:
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights_eog)

        fm_model_eog_global = eog_model() #temp model
        fm_model_eog_global.set_weights(averaged_weights)
        
        model1= 's_all_reconstructed_eog_encoded_global.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        try:
        	os.remove(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
        except:
        	pass
        fm_model_eog_global.save_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1),overwrite=True)


        client.publish("GlobalModel_EOG", payload = 'abc') #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model EOG to Topic:--> GlobalModel")

        #**********************************************************
        time.sleep(5) #pause it so that the publisher gets the Global model
        #**********************************************************

        #====================================================================


    print('---------------HERE------------------')
    #===================================================================================
    # If No more data from Publisher exit and server closed connection to the broker
    #===================================================================================
#     if(i >0 and len(all_local_model_weights_eog)==0): #loop break no message from producer
#         break
     
    
    time.sleep(8) #to receive model weights
    all_local_model_weights_gsr = list()

    for p in range(n):
    	model1= 's_all_reconstructed_gsr_encoded'+str(p+1)+'.h5'
    	OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
    	encoder_gsr = load_model(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
    	local_model_weights_gsr=encoder_gsr.get_weights()
    	scaled_weights = scale_model_weights(local_model_weights_gsr, 0.1)
    	all_local_model_weights_gsr.append(scaled_weights)

    print('Total Local Model Received for gsr:',len(all_local_model_weights_eog))
    #======================================================

    i +=1 #Next round increment

    #===================================================================
    # Publish the Global Model after Federated Averaging
    #===================================================================
    if i >0:
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights_gsr)

        fm_model_gsr_global = gsr_model() #temp model
        fm_model_gsr_global.set_weights(averaged_weights)
        
        model1= 's_all_reconstructed_gsr_encoded_global.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        try:
        	os.remove(os.path.join(OUTPUT, 'Models', 'autoencoders', model1))
        except:
        	pass
        fm_model_gsr_global.save_weights(os.path.join(OUTPUT, 'Models', 'autoencoders', model1),overwrite=True)


        client.publish("GlobalModel_GSR", payload = 'abc') #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model GSR to Topic:--> GlobalModel")

        #**********************************************************
        time.sleep(5) #pause it so that the publisher gets the Global model
        #**********************************************************

        #====================================================================


    print('---------------HERE------------------')
    #===================================================================================
    # If No more data from Publisher exit and server closed connection to the broker
    #===================================================================================
    if(i >0 and len(all_local_model_weights_gsr)==0): #loop break no message from producer
        break
        
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

i = 0
while True:
    print('---------STARTED-------------')
    print('Global Server')
    # print('Round: ',i)

    if (i>0):
    	print('waiting for performance metrics')
    	time.sleep(5)
    #====================================================
    # Global Model Performance Printing
    #====================================================

    if (i>0): #after first round of model exchange global models performance is calculated
        print('Now Collecting Local Model performance Metrics....')
        local_model_performace = list()
        while not qGSPerformance_Valence.empty():
            message = qGSPerformance_Valence.get()

            if message is None:
                continue

            msg_model_performance = message.payload.decode('utf-8')

            decodedModelPerfromance = list(json.loads(msg_model_performance).values())
            local_model_performace.append(decodedModelPerfromance)

        global_model_performance = np.array(local_model_performace)
        global_performance = np.mean(global_model_performance, axis=0)

        len_local_perfm = len(local_model_performace)
        print('Total Model Performance received:',len_local_perfm)

        if (len_local_perfm != 0):
            global_model_result.append([global_performance[1]])
            print('----------------------------------------------------')
            print('F1 score for Low Valence:',global_performance[5])
            print('F1 score for High Valence:',global_performance[6])
            
            print('----------------------------------------------------')
        else:
            break #No more data from local model

    #**********************************************************
    print('waiting for local models')
    time.sleep(10) #to receive model weights
    #**********************************************************

    #=========================================================
    # Local Model Receiving Part
    #=========================================================
    all_local_model_weights_valence = list()
    model_available=0
    
    while not qGSValence.empty():
            message = qGSValence.get()

            if message is None:
                continue

            msg_model = message.payload.decode('utf-8')
            print('msg_model is ',msg_model)
            if(msg_model=='abc'):
            	model_available=1

    if(model_available==0):
    	print("There is no local model to receoive")
    	break
    for p in range(n):
        model1= 's_all_reconstructed_valence_encoded'+str(p+1)+'.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        encoder_valence = load_model(os.path.join(OUTPUT, 'Models', 'valence', model1))
        local_model_weights_valence=encoder_valence.get_weights()
        scaled_weights = scale_model_weights(local_model_weights_valence, 0.1)
        all_local_model_weights_valence.append(scaled_weights)



    print('Total Local Model Received for Valence:',len(all_local_model_weights_valence))

    #======================================================

    i +=1 #Next round increment

    #===================================================================
    # Publish the Global Model after Federated Averaging
    #===================================================================
    if i >0:
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights_valence)

        #global_weights = EagerTensor2Numpy(averaged_weights)
        #encodedGlobalModelWeights = json.dumps(global_weights,cls=Numpy2JSONEncoder)


        fm_model_valence_global = valence_classification(global_weight_emg,global_weight_eog,global_weight_gsr) #temp model
        fm_model_valence_global.set_weights(averaged_weights)
        model1= 's_all_reconstructed_valence_encoded_global.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        try:
        	os.remove(os.path.join(OUTPUT, 'Models', 'valence', model1))
        except:
        	pass
        fm_model_valence_global.save_weights(os.path.join(OUTPUT, 'Models', 'valence', model1),overwrite=True)
        client.publish("GlobalModel_Valence", payload = 'abc') #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model to Topic:--> GlobalModel_Valence")

        #**********************************************************
        time.sleep(5) #pause it so that the publisher gets the Global model
        #**********************************************************

        #====================================================================


    print('---------------HERE------------------')
 #**********************************************************
        #arousal part
 #**********************************************************   
    
i = 0
while True:
    print('---------STARTED-------------')
    print('Global Server')
    # print('Round: ',i)

    if (i>0):
    	print('waiting for performance metrics')
    	time.sleep(5)
    #====================================================
    # Global Model Performance Printing
    #====================================================

    if (i>0): #after first round of model exchange global models performance is calculated
        print('Now Collecting Local Model performance Metrics....')
        local_model_performace = list()
        while not qGSPerformance_Arousal.empty():
            message = qGSPerformance_Arousal.get()

            if message is None:
                continue

            msg_model_performance = message.payload.decode('utf-8')

            decodedModelPerfromance = list(json.loads(msg_model_performance).values())
            local_model_performace.append(decodedModelPerfromance)

        global_model_performance = np.array(local_model_performace)
        global_performance = np.mean(global_model_performance, axis=0)

        len_local_perfm = len(local_model_performace)
        print('Total Model Performance received:',len_local_perfm)

        if (len_local_perfm != 0):
            global_model_result.append([global_performance[1]])
            print('----------------------------------------------------')
            print('F1 score for Low Arousal:',global_performance[5])
            print('F1 score for High Arousal:',global_performance[6])
            
            print('----------------------------------------------------')
        else:
            break #No more data from local model

    #**********************************************************
    print('waiting for local models')
    time.sleep(10) #to receive model weights
    #**********************************************************

    #=========================================================
    # Local Model Receiving Part
    #=========================================================
    all_local_model_weights_arousal = list()
    model_available=0
    
    while not qGSArousal.empty():
            message = qGSArousal.get()

            if message is None:
                continue

            msg_model = message.payload.decode('utf-8')
            print('msg_model is ',msg_model)
            if(msg_model=='abc'):
            	model_available=1

    if(model_available==0):
    	print("There is no local model to receoive")
    	break
    for p in range(n):
        model1= 's_all_reconstructed_arousal_encoded'+str(p+1)+'.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        encoder_arousal = load_model(os.path.join(OUTPUT, 'Models', 'arousal', model1))
        local_model_weights_arousal=encoder_arousal.get_weights()
        scaled_weights = scale_model_weights(local_model_weights_arousal, 0.1)
        all_local_model_weights_arousal.append(scaled_weights)



    print('Total Local Model Received for Arousal:',len(all_local_model_weights_arousal))

    #======================================================

    i +=1 #Next round increment

    #===================================================================
    # Publish the Global Model after Federated Averaging
    #===================================================================
    if i >0:
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights_arousal)

        #global_weights = EagerTensor2Numpy(averaged_weights)
        #encodedGlobalModelWeights = json.dumps(global_weights,cls=Numpy2JSONEncoder)


        fm_model_arousal_global = arousal_model(global_weight_emg,global_weight_eog,global_weight_gsr) #temp model
        fm_model_arousal_global.set_weights(averaged_weights)
        model1= 's_all_reconstructed_arousal_encoded_global.h5'
        OUTPUT="/home/csis/Documents/Fed-ReMECS-mqtt-main"
        try:
        	os.remove(os.path.join(OUTPUT, 'Models', 'arousal', model1))
        except:
        	pass
        fm_model_arousal_global.save_weights(os.path.join(OUTPUT, 'Models', 'arousal', model1),overwrite=True)
        client.publish("GlobalModel_Arousal", payload = 'abc') #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model to Topic:--> GlobalModel_Arousal")

        #**********************************************************
        time.sleep(5) #pause it so that the publisher gets the Global model
        #**********************************************************

        #====================================================================


    print('---------------HERE------------------')




#Global Model Result Save
# folderPath = '/home/gp/Desktop/PhD-codes/Fed-ReMECS-mqtt/Federated_Results/'
# fname_fm = folderPath +'_Global_Model' +'_'+'_results.csv'
# column_names = ['Acc', 'F1']
# global_model_result = pd.DataFrame(global_model_result,columns = column_names)
# global_model_result.to_csv(fname_fm)


print("All done, Global Server Closed.")
client.loop_stop()

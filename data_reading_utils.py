import pandas as pd
import numpy as np
import os 
import pickle 
#=================================================================================================
# Physiological data read according to the client
#=================================================================================================

##===================================================
# EEG data read from files
##===================================================
# def eeg_data(p):
#     file_eeg = '/home/gp/Desktop/MER_arin/DEAP_data/eeg_data/'+str(p)+'_data_DEAP'+'.csv'
#     print(file_eeg)
#     eeg_sig = pd.read_csv(file_eeg,sep=',', header = None, engine='python')
#     return eeg_sig

##===================================================
# 
##===================================================
def get_emg_eog_gsr_labels_data(p):
    f='data_preprocessed_python'
    physio_data_all = []
    label_data_all = []
#     file = os.path.join(r, i) //check later
    if p<=8:
        file = '/home/csis/Documents/data_preprocessed_python/s0'+str(p+1)+'.dat'
    else:
        file = '/home/csis/Documents/data_preprocessed_python/s'+str(p+1)+'.dat'
    
    with open(file, 'rb') as s_data: 
        content = pickle.load(s_data, encoding='latin1')
        physio_data_all.append(content['data'])
        label_data_all.append(content['labels'])
# for (r, d, f) in os.walk(f):
#   for i in f:
#     print(i)

    p_all = np.array(physio_data_all)
    l_all = np.array(label_data_all)
    EMG_all = p_all[:,:,34:36,:]
    EOG_all = p_all[:,:,32:34,:]
    GSR_all = p_all[:,:,36,:]
    
    print(EMG_all.shape,GSR_all.shape,len(label_data_all))
    
    return EMG_all,EOG_all,GSR_all,label_data_all

##===================================================
# 
##===================================================
def get_eog_v(i,EOG_all):
    s=EOG_all[0]
    print('shape of s',s.shape)
    t = s[i].T
    t = t[128*3:]
    t = t.reshape((-1, 128, 2))
    t_EOG = np.array(t)
    
    return t_EOG

def get_emg_v(i,EMG_all):
    s=EMG_all[0]
    
    t = s[i].T
    t = t[128*3:]
    t = t.reshape((-1, 128, 2))
    t_EMG = np.array(t)
    
    return t_EMG

def get_gsr_v(i,GSR_all):
    s=GSR_all[0]
    
    t = s[i].T
    t = t[128*3:]
    t = t.reshape((-1, 128, 1))
    t_GSR = np.array(t)
    
    return t_GSR


##===================================================
# 
##===================================================
def get_data_video(i,EMG_all,EOG_all,GSR_all,label_data_all):
    t_EOG_all = []
    t_EOG_all = get_eog_v(i,EOG_all)
    t_EMG_all = []
    t_EMG_all = get_emg_v(i,EMG_all)
    t_GSR_all = []
    t_GSR_all = get_gsr_v(i,GSR_all)
    
    y_all = []
    l=label_data_all[0]
    y = np.ones((60,1))*l[i]
    
    temp = []
    for j in y:
      y_val=[]
      for i in j:
          if i>5:
              y_val.append(1)
          else:
              y_val.append(0)
      temp.append(y_val)
            
    y_all = np.array(temp)
    y_all_concat = y_all.reshape(-1, 4)
    from keras.utils import np_utils

    y = np_utils.to_categorical(y_all_concat)


    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder

    scaler1 = StandardScaler()
    val_EMG = t_EMG_all.reshape(-1, 2)
    scaler1 = scaler1.fit(val_EMG)
    EMG = scaler1.transform(val_EMG)
    t_EMG = EMG.reshape(-1, 128, 2)

    scaler2 = StandardScaler()
    val_EOG = t_EOG_all.reshape(-1, 2)
    scaler2 = scaler2.fit(val_EOG)
    EOG = scaler2.transform(val_EOG)
    t_EOG = EOG.reshape(-1, 128, 2)

    scaler3 = StandardScaler()
    val_GSR = t_GSR_all.reshape(-1, 1)
    scaler3 = scaler3.fit(val_GSR)
    GSR = scaler3.transform(val_GSR)
    t_GSR = GSR.reshape(-1, 128, 1)
    
    return t_EMG,t_EOG,t_GSR,y

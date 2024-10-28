# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:56:13 2024

@author: 555
"""

from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold
import tensorflow.keras as keras
#import keras
import numpy as np
import pandas as pd
import tensorflow as tf 
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import pickle
from keras.layers import GlobalAveragePooling1D
import random

import os
import matplotlib



def build_model_lmsys2(drop_rate, learning_rate_value, decay_rate, decay_steps):
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(100, 2)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    #model.add(GlobalAveragePooling1D())
    
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))


    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))


    model.add(Flatten())
    
    
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(Dropout(drop_rate))

    model.add(Dense(1, activation='sigmoid'))  
    
    
    lr_scheduler = ExponentialDecay(
        initial_learning_rate=learning_rate_value,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr_scheduler)


    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model 


def build_model_lmsys_complex_v1(drop_rate, learning_rate_value, decay_rate, decay_steps):
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(100, 2)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    '''
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))  
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    '''
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(Dropout(drop_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    lr_scheduler = ExponentialDecay(
        initial_learning_rate=learning_rate_value,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr_scheduler)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model



def process_features(df):
    def parse_feature(row):
        return [list(map(float, x.strip('()').split(','))) for x in row]

    features = df.iloc[:, :-1].apply(parse_feature, axis=1)
    features = np.stack(features.values)
    return features




def train_model1(train_data, saveModel_path, saveScaler_path, pic_save_path, seed, drop_rate, learning_rate_value, decay_rate ,decay_steps):
    # 确保使用GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU for training.")
    else:
        print("No GPU found, using CPU.")
    
    model = build_model_lmsys2(drop_rate, learning_rate_value, decay_rate, decay_steps)
    #model = build_model_lmsys_complex_v1(drop_rate, learning_rate_value, decay_rate, decay_steps)

    df = pd.read_csv(train_data, header=None, dtype=str)
    df1 = df.sample(frac=1).reset_index(drop=True)  
    #print(df1.info)
    
    
   

    train_df, val_df = train_test_split(df1, test_size=0.2, random_state=seed,shuffle=True)

    train_features = process_features(train_df)
    train_labels = train_df.iloc[:, -1].values.astype(np.float32)  

    val_features = process_features(val_df)
    val_labels = val_df.iloc[:, -1].values.astype(np.float32)

    scaler = StandardScaler()

    train_features_reshaped = train_features.reshape(-1, 2)  
    train_features_reshaped = scaler.fit_transform(train_features_reshaped)
    train_features = train_features_reshaped.reshape(-1, 100, 2)  

    val_features_reshaped = val_features.reshape(-1, 2) 
    val_features_reshaped = scaler.transform(val_features_reshaped)
    val_features = val_features_reshaped.reshape(-1, 100, 2)  
    
    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    history = model.fit(
        train_features,
        train_labels,
        epochs=200,
        batch_size=32,
        callbacks=[es],
        shuffle=True,
        validation_data=(val_features, val_labels)
    )
    

    
    loss, accuracy = model.evaluate(val_features, val_labels)
    print(f'Test accuracy: {accuracy:.2f}')
    print(f'Test loss: {loss:.2f}')
    
    train_process_loss_accuracy_plot(history , pic_save_path )
    
    model.save(saveModel_path)
    with open(saveScaler_path,'wb') as f:
        pickle.dump(scaler, f)
    
    return loss,accuracy
    

def train_process_loss_accuracy_plot(history, pic_save_path):


    matplotlib.use('Agg')
    plt.figure()
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(pic_save_path + "/acc.png")
    #plt.show()
    
    
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(pic_save_path + "/loss.png")
    #plt.show()
    
    

def train_data_visualize(train_data):
  
    df = pd.read_csv(train_data, header=None)
    train_features = process_features(df)
    all_mz = []
    for i in train_features:
        for j in i:
            mz = j[0]
            all_mz.append(mz)

    print(len(all_mz))
    bins = np.arange(0,7001,500)
    counts, bin_edges = np.histogram(all_mz, bins=bins)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, counts, width=400, edgecolor='black', align='center')
    
    plt.xlabel('mz')
    plt.ylabel('number')
    plt.title('Training data distribution')
    
    plt.xticks(bin_centers, [f'{int(edge)}-{int(edge+499)}' for edge in bin_edges[:-1]], rotation=45)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    
from sklearn.metrics import accuracy_score

def roc_precision_recall_plot(test_data,model_path, scaler_path, pic_save_path):

    loaded_model = load_model(model_path)
    
    df = pd.read_csv(test_data)
    df = df.sample(frac=1).reset_index(drop=True)  
    print(df.info)

    test_features = process_features(df)
    test_labels = df.iloc[:, -1].values.astype(np.float32) 
    

    test_features_reshaped = test_features.reshape(-1, 2) 

   
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    test_features_reshaped = scaler.transform(test_features_reshaped)
  
    test_features = test_features_reshaped.reshape(-1, 100, 2)  
    
    predict_x = loaded_model.predict(test_features)
    y_true = test_labels
    y_pred = np.where(predict_x >0.3,1,0)
    acc = accuracy_score(y_true, y_pred)
    print(f'Validation acc: {acc:.2f}')

   


    matplotlib.use('Agg')
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(pic_save_path + "/roc2.png")
    #plt.show()
    
   
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve (area = %0.2f)' % ap_score)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.savefig(pic_save_path + "/precisin_recall2.png")
    #plt.show()    

    TPR_TNR_rate(y_true, predict_x, pic_save_path)

    return acc





import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_tpr_tnr(y_true, y_pred_prob, threshold):
    
    y_pred = np.where(y_pred_prob >= threshold, 1, 0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0   
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0   
    fnr = fn / (tp + fn) if (tp + fn) != 0 else 0     
    
    return tpr, tnr, fnr





def TPR_TNR_rate(y_true, predict_x, pic_save_path):
    thresholds = np.arange(0.0, 1.1,0.1)
    tpr_list = []
    tnr_list = []
    fnr_list = []
    
    for threshold in thresholds:
        tpr, tnr, fnr = calculate_tpr_tnr(y_true, predict_x, threshold)
        tpr_list.append(tpr)
        tnr_list.append(tnr)
        fnr_list.append(fnr)
    
    matplotlib.use('Agg')
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr_list, label='TPR (True Positive Rate)', color='darkorange', lw=2)
    plt.plot(thresholds, tnr_list, label='TNR (True Negative Rate)', color='navy', lw=2)
    #plt.plot(thresholds, fnr_list, label='TNR (False Negative Rate)', color='navy', lw=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and TNR vs. Threshold (0 - 1)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(pic_save_path + "/tpr_tnr_0_1.png")
    #plt.show()
    

    for i,val in enumerate(thresholds):
        print(val)
        print(f"current tpr:{tpr_list[i]}")
        print(f"current tnr:{tnr_list[i]}")
        print("  ")

        
    
    thresholds = np.arange(0.0, 0.41,0.05)
    tpr_list = []
    tnr_list = []
    fnr_list = []
    
    for threshold in thresholds:
        tpr, tnr, fnr = calculate_tpr_tnr(y_true, predict_x, threshold)
        tpr_list.append(tpr)
        tnr_list.append(tnr)
        fnr_list.append(fnr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr_list, label='TPR (True Positive Rate)', color='darkorange', lw=2)
    plt.plot(thresholds, tnr_list, label='TNR (True Negative Rate)', color='navy', lw=2)
    #plt.plot(thresholds, fnr_list, label='TNR (False Negative Rate)', color='navy', lw=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and TNR vs. Threshold   (0 - 0.4)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(pic_save_path + "/tpr_tnr_0_04.png")
    #plt.show()
    
    
    
    for i,val in enumerate(thresholds):
        print(val)
        print(f"current tpr:{tpr_list[i]}")
        print(f"current tnr:{tnr_list[i]}")
        print("  ")
        print("  ")
        print("  ")
    
    

    
def process_features_fortesting(df):
    def parse_feature(row):
        return [list(map(float, x.strip('()').split(','))) for x in row]

    features = df.iloc[:, :].apply(parse_feature, axis=1)
    features = np.stack(features.values)
    return features 




def predict_model(model_path, scaler_path, test_data):
    
    loaded_model = load_model(model_path)
    df = pd.read_csv(test_data)
    test_features = process_features_fortesting(df)

    test_features_reshaped = test_features.reshape(-1, 2)  
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    test_features_reshaped = scaler.transform(test_features_reshaped)
    test_features = test_features_reshaped.reshape(-1, 100, 2)  
    
    predict_x = loaded_model.predict(test_features)
    print(predict_x)
    
    


def train_start(seed, learning_rate, drop_rate, decay_rate, decay_steps):
    
    folder_name = "seed"+str(seed)+"_lr"+str(learning_rate)+"_drooput"+str(drop_rate)+"_decayRate"+str(decay_rate)+"_decaySteps"+str(decay_steps)
    save_model_name="model_"+"seed"+str(seed)+"_lr"+str(learning_rate)+"_drooput"+str(drop_rate)+"_decayRate"+str(decay_rate)+"_decaySteps"+str(decay_steps)
    save_scaler_name = "scaler_"+"seed"+str(seed)+"_lr"+str(learning_rate)+"_drooput"+str(drop_rate)+"_decayRate"+str(decay_rate)+"_decaySteps"+str(decay_steps)
    
    
    default_model_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name+"/"
    if not os.path.exists(default_model_path):
        os.mkdir(default_model_path)

    #traindata_ms2_data3_4_5_6_7_13_mix_ms1data    train_ms2_data3_4_5_6_7_13.csv
    test_loss, test_accuracy = train_model1(train_data="D:/gitclone/denoise/data/train/train_ms2_data3_4_5_6_7_13_1.csv",
                 saveModel_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name + "/"+save_model_name+".keras",
                 saveScaler_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name + "/"+save_scaler_name +".pkl",
                 pic_save_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name,
                 seed=seed, drop_rate=drop_rate, learning_rate_value = learning_rate, decay_rate = decay_rate, decay_steps = decay_steps)
    
    #valiation set
    
    val_acc = roc_precision_recall_plot(model_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name + "/"+save_model_name+".keras", 
                  test_data = "D:/gitclone/denoise/data/train/validate_ms2_hetangti_untilscan52.csv", 
                  scaler_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name + "/"+save_scaler_name +".pkl",
                  pic_save_path = "D:/gitclone/denoise/data/ms2_model/"+folder_name)
    
    #val_acc =0 
    
    return test_loss, test_accuracy, val_acc
    
    
    
    




#train_data_visualize(train_data="D:/gitclone/denoise/data/train/train_ms2_data3_4_5_6_7.csv")



seed_list = [42,3407]
drop_out_list = [0.1, 0.3, 0.5, 0.9]
decay_rate_list = [0.92,0.96]
decay_steps_list  = [500,1000]
learning_rate_list = [0.01,0.005]


results = []


for seed in seed_list:
    for learning_rate in learning_rate_list:
        for drop_rate in drop_out_list:
            for decay_rate in decay_rate_list:
                for decay_steps in decay_steps_list:
                        setting_para = "seed"+str(seed)+"_lr"+str(learning_rate)+"_drooput"+str(drop_rate)+"_decayRate"+str(decay_rate)+"_decaySteps"+str(decay_steps)
                        test_loss, test_accuracy, val_acc = train_start(seed=seed, learning_rate = learning_rate, drop_rate = drop_rate, decay_rate = decay_rate, decay_steps = decay_steps)
                        results.append({
                            'Setting': setting_para,
                            'Test_Loss': test_loss,
                            'Test_Accuracy': test_accuracy,
                            'Validation_Accuracy': val_acc
                        })
                        df = pd.DataFrame(results)
                        df.to_csv('D:/gitclone/denoise/data/ms2_model/model_training_result_temp_0928.csv', index=False, mode='a')
                        

df = pd.DataFrame(results)
df.to_csv('D:/gitclone/denoise/data/ms2_model/model_training_results_final_0928.csv', index=False, mode='a')




'''
roc_precision_recall_plot(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
              test_data = "D:/gitclone/denoise/data/train/train_ms2_data3_4_5_6_7.csv", 
              scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
              pic_save_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000")

'''


'''


roc_precision_recall_plot(model_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
              test_data = "D:/gitclone/denoise/data/train/traindata_data13.csv", 
              scaler_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
              pic_save_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1")


roc_precision_recall_plot(model_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
              test_data = "D:/gitclone/denoise/data/train/validate_ms2_hetangti_untilscan52.csv", 
              scaler_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
              pic_save_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1")


roc_precision_recall_plot(model_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500/model_seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500.keras", 
              test_data = "D:/gitclone/denoise/data/train/validate_ms2_hetangti_untilscan52.csv", 
              scaler_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500/scaler_seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500.pkl",
              pic_save_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1")

'''
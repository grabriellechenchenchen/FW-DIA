# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:14:52 2024

@author: 555
"""

import pymzml
import pyopenms
from interval import Interval
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pyopenms import MSExperiment, MzMLFile


def raw_file_split(mzmlFile_name, mzmlFile_path, scanid,mzStart,mzEnd):  
    mzml_file = mzmlFile_path +'/'+mzmlFile_name
    window_length=100
    run = pymzml.run.Reader(mzml_file)
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file, exp)
    spec_count = 0
    peaks_group = []
    for spectrum in exp:
        peaks=[]
        spec_count += 1
        if spec_count == scanid:  
            peaks = [(peak.getMZ(), peak.getIntensity()) for peak in spectrum if peak.getMZ() in Interval(mzStart,mzEnd)]
            for i in range(0, len(peaks), window_length):
                chunk = peaks[i:i + window_length]
                peaks_group.append(chunk)   
   
    #print(len(peaks_group))
    return peaks_group



def process_features(df):
    def parse_feature(row):
        return [list(map(float, x.strip('()').split(','))) for x in row]

    features = df.iloc[:, :-1].apply(parse_feature, axis=1)
    features = np.stack(features.values)
    return features




def raw_file_start(model_path, mzmlFile_name, mzmlFile_path,scanStart,scanEnd,mzStart,mzEnd, scaler_path):
    loaded_model = load_model(model_path)
    result_df = pd.DataFrame(columns=['scanID', 'Index','chunk', 'predict_x', 'prediction_class'])

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    for scanid in range(scanStart,scanEnd):
        peaks_group = raw_file_split(mzmlFile_name, mzmlFile_path, scanid,mzStart,mzEnd)
        dff = []
        for i,chunk in enumerate(peaks_group):
            if len(chunk) ==100:
                df = pd.DataFrame(chunk)
                #scaler = StandardScaler()
                test_features = np.stack(df.values)
                
                test_features_reshaped = test_features.reshape(-1, 2)  

                test_features_reshaped = scaler.transform(test_features_reshaped)
                test_features = test_features_reshaped.reshape(-1, 100, 2)  
                
                predict_x = loaded_model.predict(test_features)
                prediction_class = np.where(predict_x > 0.3,1,0)
                
                result_df = result_df.append({
                    'scanID': scanid,
                    'Index': i,
                    'chunk': [chunk[0][0],chunk[-1][0]],
                    'predict_x': predict_x,
                    'prediction_class': prediction_class
                }, ignore_index=True) 
                
    result_df.to_csv(mzmlFile_path +'/preduction_result_threshold_scan150_onlyms2Model_th02.csv', index=False)        



def prediction_result_visualization(mzmlFile_path,mz_1,mz_2): 
    prediction_result_file = mzmlFile_path +'/preduction_result_threshold_scan125_onlyms2Model_th02.csv'
    df = pd.read_csv(prediction_result_file) 

    df['chunk'] = df['chunk'].apply(eval)
    df['prediction_class'] = df['prediction_class'].apply(lambda x: int(x[2]))
    print(df['prediction_class'])
    plt.figure(figsize=(18,10))
    
    for idx, row in df.iterrows():
        start, end = row['chunk']
        if start > mz_1 and end < mz_2:
            if row['prediction_class'] == 1:
                plt.fill_between([start, end], 1, color='lightblue', alpha=0.5, step='pre')

    plt.xlim(df['chunk'].apply(lambda x: x[0]).min(), df['chunk'].apply(lambda x: x[1]).max())
    plt.xlim(mz_1,mz_2)
    plt.ylim(0, 1)

    plt.title('Regions with Prediction Class 1')
    plt.xlabel('Chunk Range')
    plt.ylabel('Prediction Class')  

    plt.show()
    

import ast
def mzml_denoise(mzmlFile_name, mzmlFile_path, preduction_result_path, scanid, output_file): 
    exp = MSExperiment()
    mzml = MzMLFile()
    mzml.load(mzmlFile_path +'/'+ mzmlFile_name, exp)
    df = pd.read_csv(preduction_result_path)
    df['chunk'] = df['chunk'].apply(ast.literal_eval)
    label_0 = df[(df['prediction_class'].apply(lambda x: int(x[2]))) == 0]
    
    print(len(label_0))
    

    for i,spectrum in enumerate(exp):
        if spectrum.getMSLevel() == 1 or spectrum.getMSLevel() == 2:
            if i+1 == scanid:
                mz_array, intensity_array = spectrum.get_peaks()
                #print(mz_array[10])
                #print(f"origal: {intensity_array[10]}") 
                
                noise_ranges = label_0['chunk'].tolist()
                #print( type(noise_ranges[0] ))
                modified_intensity = []

                for mz, intensity in zip(mz_array, intensity_array):
                    is_noise = False
                    for noise_range in noise_ranges:

                        if noise_range[0] <= mz <= noise_range[1]:
                            #print(f"hhh: {noise_range[0]}")
                            is_noise = True
                            break
                    #print(is_noise)
                    if is_noise:
                        modified_intensity.append(0)
                    else:
                        modified_intensity.append(intensity)

                spectrum.set_peaks((mz_array, modified_intensity))
                exp[i] = spectrum
                #print(f"modify: {modified_intensity[10]}") 
                print(len(modified_intensity))

    mzml.store(output_file, exp)
    
    
    
    for i,spectrum in enumerate(exp):
        if i+1 == scanid:
            mz_array, intensity_array = spectrum.get_peaks()
            print(f"3: {intensity_array[1]}") 
    
    
   # mzml.store("D:/gitclone/denoise/data/mzmlfile/66/filter3_centroid.mzML", modified_exp)



def mzml_denoise_lm(mzmlFile_name, mzmlFile_path, preduction_result_path, scanid, output_file):
    exp = MSExperiment()
    mzml = MzMLFile()
    mzml.load(mzmlFile_path + '/' + mzmlFile_name, exp)
    df = pd.read_csv(preduction_result_path)
    
    label_0 = df[(df['prediction_class'].apply(lambda x: x[0]) == 0) & (df['scanid'] == scanid)]

    for i, spectrum in enumerate(exp):
        if spectrum.getMSLevel() == 1 or spectrum.getMSLevel() == 2:
            if i + 1 == scanid:
                mz_array, intensity_array = spectrum.get_peaks()

                noise_ranges = label_0['chunk'].tolist()
                modified_intensity = []

                for mz, intensity in zip(mz_array, intensity_array):
                    is_noise = False
                    for noise_range in noise_ranges:
                        if noise_range[0] <= mz <= noise_range[1]:
                            is_noise = True
                            break
                    if is_noise:
                        modified_intensity.append(0)
                    else:
                        modified_intensity.append(intensity)

                spectrum.set_peaks((mz_array, modified_intensity))
                exp[i] = spectrum

    mzml.store(output_file, exp)



def topFD_flashdeconv_union(flashd_mass_file, fd_msalign_file,scanID):
    spec_ms1_df = pd.read_csv(flashd_mass_file,sep='\t')
    spec_ms1_df['MonoisotopicMass'] = spec_ms1_df['MonoisotopicMass'].astype(float)   #mass
    spec_ms1_df['MinCharge'] = spec_ms1_df['MinCharge'].astype(float)    #charge
    spec_ms1_df['MaxCharge'] = spec_ms1_df['MaxCharge'].astype(float)
    spec_ms1_df['ScanNum'] = spec_ms1_df['ScanNum']
    spec_ms1_df['Index'] = spec_ms1_df['Index'].astype(float)
    this_scan_df = spec_ms1_df[(spec_ms1_df['ScanNum'] ==2)]


    flash_align_fd = []
    for  _, rt_row in this_scan_df.iterrows():
        flash_monoMass = float(rt_row['MonoisotopicMass'])
        flash_scanNum = int(float(rt_row['ScanNum']))
        flash_charge_min = float(rt_row['MinCharge'])
        flash_charge_max = float(rt_row['MaxCharge'])
        flash_featureIndex = int(float(rt_row['Index']))
        #print(flash_scanNum)
        thisScanInfo = []
        with open(fd_msalign_file,'r') as infile:
            read_this_scan = False
            thieOne = False
            for line in infile:
                if 'SCANS='+str(flash_scanNum) in line.strip():
                    read_this_scan =True
                    
                if read_this_scan:
                    if line.strip() == 'LEVEL=1':
                        thieOne = True
                        continue
                    if line.strip() == 'END IONS':
                        thieOne = False
                        read_this_scan = False 
                        
                        for i in thisScanInfo:
                            if i[0] in Interval(flash_monoMass-20,flash_monoMass+20):
                                if i[1] in Interval(flash_charge_min,flash_charge_max):
                                    
                                    flash_align_fd.append(flash_featureIndex)
                                    break
                        
                        thisScanInfo = []
                    if thieOne:
                        line_info = line.strip()
                        mass = float(line_info.split('\t')[0])
                        charge = float(line_info.split('\t')[2])
                        thisScanInfo.append((mass,charge))

    print(len( flash_align_fd))




from pymzml.run import Reader
import math
from tqdm import tqdm
import time
def mzml_batch_predict(model_path, mzmlFile_name, mzmlFile_path,scanStart,scanEnd,mzStart,mzEnd, scaler_path, predict_threshold):
    loaded_model = load_model(model_path)
    result_df = pd.DataFrame(columns=['scanID', 'Index','chunk', 'predict_x', 'prediction_class'])

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    all_chunks = []
    for scanid in tqdm(range(scanStart,scanEnd+1)):
        peaks_group = raw_file_split(mzmlFile_name, mzmlFile_path, scanid, mzStart,mzEnd)
        dff = []
        for i,chunk in enumerate(peaks_group):
            if len(chunk) ==100:
                df = pd.DataFrame(chunk)
                test_features = np.stack(df.values)
                test_features_reshaped = test_features.reshape(-1, 2)
                all_chunks.append((scanid, i, [chunk[0][0], chunk[-1][0]], test_features_reshaped))


    if all_chunks:
        all_test_features = np.array([chunk[3] for chunk in all_chunks])
        all_test_features_reshaped = all_test_features.reshape(-1, 2)
        
        all_test_features_reshaped = scaler.transform(all_test_features_reshaped)
        all_test_features = all_test_features_reshaped.reshape(-1, 100, 2)

        all_predict_x = loaded_model.predict(all_test_features)
        all_prediction_class = np.where(all_predict_x > predict_threshold, 1, 0)

        new_rows = []
        for (scanid, i, chunk, _), predict_x, prediction_class in zip(all_chunks, all_predict_x, all_prediction_class):
            new_row = {
                'scanID': scanid,
                'Index': i,
                'chunk': chunk,
                'predict_x': predict_x,
                'prediction_class': prediction_class
            }
            new_rows.append(new_row)
        
        new_df = pd.DataFrame(new_rows)
        result_df = pd.concat([result_df, new_df], ignore_index=True)

    
    result_df.to_csv(mzmlFile_path + '/preduction_result_batchprocess_.csv', index=False)

    

def adjust_intensities(mzmlFile_path, mzmlFile_name, prediction_result_path, output_file):
    try:
        exp = MSExperiment()
        mzml = MzMLFile()
        mzml.load(mzmlFile_path + '/' + mzmlFile_name, exp)
        df = pd.read_csv(prediction_result_path)

        try:
            df['chunk'] = df['chunk'].apply(ast.literal_eval)
            df['scanID'] = df['scanID'].astype(int)  
            label_0 = df[(df['prediction_class'].apply(lambda x: int(x[1]))) == 0]
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing input data: {e}")
            return 

        for i, spectrum in enumerate(exp):
            print(i)
            if spectrum.getMSLevel() in [1, 2]:  
                scanID = i + 1  
                
                #if scanID == 1:
                    #print(scanID)
                try:
                    scan_data = label_0[label_0['scanID'] == scanID]
                    noise_ranges = scan_data['chunk'].tolist()
                except KeyError:
                    print("Skip if scanID not found")
                    continue 

                if not noise_ranges:
                    print("Skip if no noise ranges for this scan")
                    continue 

                mz_array, intensity_array = spectrum.get_peaks()
                
                '''
                modified_intensity = []
                #print(mz_array[10])
                #print(f"origal: {intensity_array[2]}") 

                for mz, intensity in zip(mz_array, intensity_array):
                    is_noise = False
                    for noise_range in noise_ranges:
                        if noise_range[0] <= mz <= noise_range[1]:
                            is_noise = True
                            break
                    modified_intensity.append(0 if is_noise else intensity)
                
                spectrum.set_peaks((mz_array, modified_intensity))
                '''
                
                mz_array = np.array(mz_array)
                intensity_array = np.array(intensity_array)

                for noise_range in noise_ranges:
                    mask = (mz_array >= noise_range[0]) & (mz_array <= noise_range[1])
                    intensity_array[mask] = 0

                spectrum.set_peaks((mz_array, intensity_array.tolist()))
                
                
                exp[i] = spectrum
                #print(f"modify: {modified_intensity[2]}")

        mzml.store(output_file, exp)
        print(f"Modified mzML file saved to: {output_file}")
    except FileNotFoundError:
        print(f"Error: File not found. Check paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

   
def predict_signal_noise_with_model(model_path, scaler_path, mzmlFile_name, mzmlFile_path):
    mzml_file = mzmlFile_path +'/'+mzmlFile_name
    run = pymzml.run.Reader(mzml_file)
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file, exp)
    scanStart = 1
    scanEnd = Reader(mzml_file).get_spectrum_count()
    print(scanEnd)
    mz_array, intensity_array = exp[0].get_peaks()
    mzStart = math.floor(mz_array[0])
    mzEnd = math.ceil(mz_array[-1])
    predict_threshold = 0.1
    start_time = time.time()
    mzml_batch_predict(model_path, mzmlFile_name, mzmlFile_path,scanStart,scanEnd,mzStart,mzEnd, scaler_path, predict_threshold)
    
    prediction_result_path = mzmlFile_path + '/preduction_result_batchprocess_.csv'
    output_file = mzmlFile_path + '/denoised_ms_edit.mzML'
    adjust_intensities(mzmlFile_path, mzmlFile_name, prediction_result_path, output_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"run time: {elapsed_time:.2f} 秒")







import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def before_after_model_peak_intensity_distribution(before_profile_mzml, after_profile_mzml):
    exp_before = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(before_profile_mzml, exp_before)
    before_intensity = []
    for i, spectrum in enumerate(exp_before):
        if i == 104:
            mz_array, intensity_array = spectrum.get_peaks()
            intensity_array = [x for x in intensity_array if x !=0]
            before_intensity.extend(intensity_array)
    
    exp_after = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(after_profile_mzml, exp_after)
    after_intensity = []
    for i, spectrum in enumerate(exp_after):
        if i== 104:
            mz_array, intensity_array = spectrum.get_peaks()
            intensity_array = [x for x in intensity_array if x !=0]
            after_intensity.extend(intensity_array)
            
    print(len(before_intensity))
    print(len(after_intensity))
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    sns.histplot(before_intensity, kde=False, bins=30, ax=ax1, log=True, color = 'blue')
    sns.histplot(after_intensity, kde=False, bins=30, ax=ax2, log=True, color = 'green')
    #ax1.set_title('Original Data')
    #ax2.set_title('Denoised Data')
    #ax1.xlabel('')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    '''    
    plt.figure(figsize=(10, 6))
    sns.histplot(before_intensity, kde=False, bins=30, log=True, color = 'blue')
    sns.histplot(after_intensity, kde=False, bins=30, log=True, color = 'green')
    plt.show()

    '''
    data = {
        'Type': ['original data'] * len(before_intensity) + ['denoised data'] * len(after_intensity),
        'Intensity': before_intensity + after_intensity
    }
    df = pd.DataFrame(data)
    
    
    plt.figure(figsize=(10, 6))
    #sns.violinplot(x='Type', y='Intensity', data=df, palette=['skyblue', 'salmon'], kernel='statlinear')
    #sns.set(style="whitegrid")
    sns.boxplot(x='Type', y='Intensity',data=df,color='white' )
    sns.stripplot(x='Type', y='Intensity', data=df, color = 'black',alpha=0.3 ,jitter=True)
    
    plt.yscale('log')  
    plt.ylabel('intensity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    '''
    
'''
before_after_model_peak_intensity_distribution(before_profile_mzml = "D:/gitclone/denoise/data/predict/wugedanbai/before/func1_centroid.mzML", 
                                               after_profile_mzml = "D:/gitclone/denoise/data/predict/wugedanbai/denoise/denoised_centroid.mzML")    

  
predict_signal_noise_with_model(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
                                scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl", 
                                mzmlFile_name = "20230727RPLCribosomalproteinCV2070Vscantime5sS1_func1.mzML", 
                                mzmlFile_path = "D:/gitclone/denoise/data/predict")





predict_signal_noise_with_model(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
                                scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl", 
                                mzmlFile_name = "202307215mixproteinMSeCV70105V_func1.mzML", 
                                mzmlFile_path = "D:/gitclone/denoise/data/predict/wugedanbai")




predict_signal_noise_with_model(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
                                scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl", 
                                mzmlFile_name = "hetangti_profile.mzML", 
                                mzmlFile_path = "D:/gitclone/denoise/data/original_005_model/hetngti1001")


predict_signal_noise_with_model(model_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_MS2（未加data1）_MS1一起/seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000/model_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="wugedanbai_ms2_profile_try2.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/original_005_model/wugedanbai1002_ms2",
               scaler_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_MS2（未加data1）_MS1一起/seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000/scaler_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.pkl")


'''

predict_signal_noise_with_model(model_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_MS2（未加data1）_MS1一起/seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000/model_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="hetangti_ms2_profile_try2.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/original_005_model/hetangti1002_ms2",
               scaler_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_MS2（未加data1）_MS1一起/seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000/scaler_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.pkl")


'''

exp = MSExperiment()
mzml = MzMLFile()
mzml.load("D:/gitclone/denoise/data/predict/denoised_ms.mzML", exp)
for i,spectrum in enumerate(exp):
    if i+1 == 1:
        mz_array, intensity_array = spectrum.get_peaks()
        print(f"3: {intensity_array[3]}") 
'''



'''

raw_file_start(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras", 
               scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl", 
               mzmlFile_name = "20230727RPLCribosomalproteinCV2070Vscantime5sS1_func1.mzML", 
               mzmlFile_path = "D:/gitclone/denoise/data/predict",
               scanStart=150,scanEnd=151,mzStart=49,mzEnd=3000)



'''











'''

raw_file_start(model_path = "D:/gitclone/denoise/data/model/0919_model_3and5and1and426713.keras",
               mzmlFile_name ="20230727RPLCribosomalproteinCV2070Vscantime5sS1_func1.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/13",
               scaler_path = "D:/gitclone/denoise/data/model/0919_model_3and5and1and426713_scaler.pkl",
               scanStart=132,scanEnd=133,mzStart=50,mzEnd=3000)




raw_file_start(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="20230727RPLCribosomalproteinCV2070Vscantime5sS1_func1.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/13",
               scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
               scanStart=132,scanEnd=133,mzStart=50,mzEnd=3000)




raw_file_start(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="202307215mixproteinMSeCV70105V_func1.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/2",
               scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
               scanStart=98,scanEnd=99,mzStart=5000,mzEnd=6000)

'''



#prediction_result_visualization(mzmlFile_path =  "D:/gitclone/denoise/data/mzmlfile/2",mz_1=5000,mz_2=6000)


'''
raw_file_start(model_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="20230727RPLCribosomalproteinCV2070Vscantime5sS1_func2.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_13",
               scaler_path = "D:/gitclone/denoise/data/ms2_model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000_try1/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
               scanStart=8,scanEnd=9,mzStart=400,mzEnd=1600)

'''


#"D:\gitclone\denoise\data\ms2_model\新建文件夹_仅含MS2数据进行训练\seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500\model_seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500.keras"
#"D:\gitclone\denoise\data\ms2_model\新建文件夹_仅含MS2数据进行训练\seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500\scaler_seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500.pkl"


#"D:\gitclone\denoise\data\ms2_model\新建文件夹_MS2（未加data1）_MS1一起\seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000\model_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.keras"
#"D:\gitclone\denoise\data\ms2_model\新建文件夹_MS2（未加data1）_MS1一起\seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000\scaler_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.pkl"
'''
raw_file_start(model_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_仅含MS2数据进行训练/seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500/model_seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500.keras",
               mzmlFile_name ="20230727RPLCribosomalproteinCV2070Vscantime5sS1_func2.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_13",
               scaler_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_仅含MS2数据进行训练/seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500/scaler_seed42_lr0.01_drooput0.1_decayRate0.96_decaySteps500.pkl",
               scanStart=125,scanEnd=126,mzStart=588,mzEnd=624)



raw_file_start(model_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_MS2（未加data1）_MS1一起/seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000/model_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="20230727RPLCribosomalproteinCV2070Vscantime5sS1_func2.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_13",
               scaler_path = "D:/gitclone/denoise/data/ms2_model/新建文件夹_MS2（未加data1）_MS1一起/seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000/scaler_seed42_lr0.01_drooput0.3_decayRate0.96_decaySteps1000.pkl",
               scanStart=125,scanEnd=126,mzStart=900,mzEnd=1000)


raw_file_start(model_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/model_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.keras",
               mzmlFile_name ="20230727RPLCribosomalproteinCV2070Vscantime5sS1_func2.mzML" , 
               mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_13",
               scaler_path = "D:/gitclone/denoise/data/model/seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000/scaler_seed42_lr0.005_drooput0.1_decayRate0.96_decaySteps1000.pkl",
               scanStart=125,scanEnd=126,mzStart=900,mzEnd=1000)
'''








#prediction_result_visualization(mzmlFile_path =  "D:/gitclone/denoise/data/mzmlfile/FUNC2_13",mz_1=588,mz_2=624)
#prediction_result_visualization(mzmlFile_path =  "D:/gitclone/denoise/data/mzmlfile/FUNC2_13",mz_1=1300,mz_2=1500)

'''
mzml_denoise(mzmlFile_name = "func1_centroid.mzML", 
             mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/10_testing/hetangti/topfd", 
             preduction_result_path =  "D:/gitclone/denoise/data/mzmlfile/13/preduction_result_threshold_scan132_tr03.csv", 
             scanid = 132, 
             output_file ="D:/gitclone/denoise/data/mzmlfile/10_testing/0923/hetangti_model_centroid_0923.mzML")

'''




'''
topFD_flashdeconv_union(fd_msalign_file="D:/gitclone/denoise/data/mzmlfile/10_testing/0923/去噪后/hetangti_model_centroid_0923_131_133_file/hetangti_model_centroid_0923_131_133_ms1.msalign", 
                        flashd_mass_file="D:/gitclone/denoise/data/mzmlfile/10_testing/0923/去噪后/hetangti_model_centroid_0923_131_133_ms1.tsv",
                        scanID=132)

topFD_flashdeconv_union(fd_msalign_file="D:/gitclone/denoise/data/mzmlfile/10_testing/0923/20230727RPLCribosomalproteinCV2070Vscantime5sS1_file/20230727RPLCribosomalproteinCV2070Vscantime5sS1_ms1.msalign", 
                        flashd_mass_file="D:/gitclone/denoise/data/mzmlfile/10_testing/0923/20230727RPLCribosomalproteinCV2070Vscantime5sS1_ms1.tsv",
                        scanID=132)
'''





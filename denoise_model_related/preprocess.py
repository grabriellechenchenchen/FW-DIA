# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:56:30 2024

@author: 555
"""


import os
import subprocess
import wx
import sys
import shutil
from pymzml.run import Reader
import pandas as pd
import csv
import json
import math
import pandas as pd
import shutil


               
def Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num,end_scan_num):

    default_flashdeconvOutput_path = mzmlFile_path + '/flashdeconv_default_output/'
    if not os.path.exists(default_flashdeconvOutput_path):
        os.mkdir(default_flashdeconvOutput_path)
        
    MS1_mzml= mzmlFile_path +'/'+mzmlFile_name
    featureMass_file=default_flashdeconvOutput_path +'mass.tsv'
    spectrumMass_file=default_flashdeconvOutput_path +'specmass_ms1.tsv'
    filter_file = default_flashdeconvOutput_path +'falshdeconv_filter.csv'

    
    flashdeconv_path = 'D:\\Software\\OpenMS-3.0.0\\bin'
    flashdeconv_default_running_cmd = 'FLASHDeconv -in {} -out {} -out_spec {}'.format(MS1_mzml,featureMass_file, spectrumMass_file)
    
    ret = subprocess.run(flashdeconv_default_running_cmd, cwd = default_flashdeconvOutput_path)
    
    df = pd.read_csv(spectrumMass_file, sep='\t')
    filtered_df = df[(df['ScanNum'] >= start_scan_num) & (df['ScanNum'] <= end_scan_num) & (df['PeakCount'] > 2) & (df['MaxCharge'] > 2)]
    filtered_df = filtered_df.drop('Unnamed: 21', axis=1)
    filtered_df.to_csv(filter_file, index=False)




def Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num,end_scan_num):

    default_flashdeconvOutput_path = mzmlFile_path + '/flashdeconv_default_output/'
    if not os.path.exists(default_flashdeconvOutput_path):
        os.mkdir(default_flashdeconvOutput_path)
        
    MS1_mzml= mzmlFile_path +'/'+mzmlFile_name
    featureMass_file=default_flashdeconvOutput_path +'mass.tsv'
    spectrumMass_file=default_flashdeconvOutput_path +'specmass_ms1.tsv'
    spectrumMass_file_2 = default_flashdeconvOutput_path +'specmass_ms2.tsv'
    filter_file = default_flashdeconvOutput_path +'falshdeconv_filter.csv'

    
    flashdeconv_path = 'D:\\Software\\OpenMS-3.0.0\\bin'
    flashdeconv_default_running_cmd = 'FLASHDeconv -in {} -out {} -out_spec {} {}'.format(MS1_mzml,featureMass_file, spectrumMass_file,spectrumMass_file_2 )
    
    ret = subprocess.run(flashdeconv_default_running_cmd, cwd = default_flashdeconvOutput_path)
    
    df = pd.read_csv(spectrumMass_file_2, sep='\t')
    filtered_df = df[(df['ScanNum'] >= start_scan_num) & (df['ScanNum'] <= end_scan_num) & (df['PeakCount'] > 3) & (df['MaxCharge'] > 0)]
    filtered_df = filtered_df.drop('Unnamed: 28', axis=1)
    filtered_df.to_csv(filter_file, index=False)




def topFd_run():
   
    return 0 




def topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num,end_scan_num):

    flash_fd_intersec = []
    count = 0 
    #flash default running
    flashd_mass_file = mzmlFile_path +'/flashdeconv_default_output/falshdeconv_filter.csv'

    spec_ms1_df = pd.read_csv(flashd_mass_file,sep=',')
    spec_ms1_df['MonoisotopicMass'] = spec_ms1_df['MonoisotopicMass'].astype(float)   #mass
    spec_ms1_df['MinCharge'] = spec_ms1_df['MinCharge'].astype(float)    #charge
    spec_ms1_df['MaxCharge'] = spec_ms1_df['MaxCharge'].astype(float)
    spec_ms1_df['ScanNum'] = spec_ms1_df['ScanNum'].astype(float)
    spec_ms1_df['Index'] = spec_ms1_df['Index'].astype(float)

    
    fd_msalign_file = mzmlFile_path + '/topfd/func2_centroid_ms1.msalign'
    flash_align_fd = []
    for  _, rt_row in spec_ms1_df.iterrows():
        flash_monoMass = float(rt_row['MonoisotopicMass'])
        flash_scanNum = int(float(rt_row['ScanNum']))
        flash_charge_min = float(rt_row['MinCharge'])
        flash_charge_max = float(rt_row['MaxCharge'])
        flash_featureIndex = int(float(rt_row['Index']))
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
                            if i[0] in Interval(flash_monoMass-5,flash_monoMass+5):
                                if i[1] in Interval(flash_charge_min,flash_charge_max):
                                   
                                    flash_align_fd.append(flash_featureIndex)
                                    break
                        
                        thisScanInfo = []
                    if thieOne:
                        line_info = line.strip()
                        mass = float(line_info.split('\t')[0])
                        charge = float(line_info.split('\t')[2])
                        thisScanInfo.append((mass,charge))

    new_csv_file = mzmlFile_path +'/flashdeconv_default_output/manual_check.csv'
    shutil.copy(flashd_mass_file, new_csv_file)
    df = pd.read_csv(new_csv_file)
    df['label'] = 0
    df.loc[df['Index'].isin(flash_align_fd), 'label'] = 1
    df.to_csv(new_csv_file, index=False) 
    



def label_1(mzmlFile_name, mzmlFile_path):

    default_flashdeconvOutput_path = mzmlFile_path + '/flashdeconv_default_output/'
    manuallycheck_file = default_flashdeconvOutput_path +'manual_check.csv'
    label1_file = default_flashdeconvOutput_path +'manual_label.csv'
    df = pd.read_csv(manuallycheck_file, sep=',')
    if 'label' in df.columns:
        filtered_df = df[(df['label'] ==1)]
        filtered_df.to_csv(label1_file, index=False)
    else:
        print("Error: 'label' column not found in the dataframe.")
    
    
    

import pymzml
import pyopenms
from interval import Interval
def raw_file_split(mzmlFile_name, mzmlFile_path,scan_id, window_length=100): 
 
    mzml_file = mzmlFile_path +'/'+mzmlFile_name

    run = pymzml.run.Reader(mzml_file)
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file, exp)
    spec_count = 0
    peaks_group = []
    for spectrum in exp:
        spec_count += 1
        if spec_count == scan_id:
            peaks = [(peak.getMZ(), peak.getIntensity()) for peak in spectrum]
            for i in range(0, len(peaks), window_length):
                chunk = peaks[i:i + window_length]
                peaks_group.append(chunk)   
    return peaks_group



def filter_chunks(scan_id, peaks_group, label_file, output_file,output_label_0_file):
    df = pd.read_csv(label_file)
    labels = df[(df['ScanNum'] ==scan_id)]
    filtered_chunks = []
    chunks_0 = []
    
    # Iterate over the chunks
    for chunk in peaks_group:
        contains_range = False

        # Iterate over the labels
        for index, row in labels.iterrows():
            start_mz = row['RepresentativeMzStart']
            end_mz = row['RepresentativeMzEnd']

            # Check if any m/z value in the chunk falls within the range
            if any(start_mz <= mz <= end_mz for mz, _ in chunk):
                contains_range = True
                break

        # If the chunk contains the desired m/z range, add it to the output list
        if contains_range:
            # Add the chunk to the output list with an additional label column
            chunk_with_label = chunk + [1]  # Add a tuple with a single element (1) as the label
            filtered_chunks.append(chunk_with_label)
        '''
        else:
            #If the chunk doesnt contain
            chunk_label_0 = chunk + [0]
            chunks_0.append(chunk_label_0 )
        '''

    # Convert the output list to a Pandas DataFrame
    df = pd.DataFrame(filtered_chunks)
    df.to_csv(output_file, mode='a',header=False, index=False)
    
    '''
    df_0 = pd.DataFrame(chunks_0)
    df_0.to_csv(output_label_0_file, mode='a',header=False, index=False)
    '''


def extract_label1_chunks(mzmlFile_name, mzmlFile_path):
 
    # Extract the unique values from the ScanNum column
    label_file = mzmlFile_path + '/flashdeconv_default_output/' +'manual_label.csv'  #_recheck
    output_file = mzmlFile_path + '/flashdeconv_default_output/' +'train_1.csv'
    output_label_0_file = mzmlFile_path + '/flashdeconv_default_output/' +'train_0_.csv'
    falsh_file = mzmlFile_path + '/flashdeconv_default_output/specmass_ms2.tsv'
    output_label_0_filter_file = mzmlFile_path + '/flashdeconv_default_output/train_0_flashdeconv_dontknow_'
    df = pd.read_csv(label_file)
    scan_nums = df['ScanNum'].unique()
    scan_nums = sorted(scan_nums)
    scan_nums = [64,97,115,122,125,129,135,139,146]
    for scan_id in scan_nums:
        print(scan_id)
        peaks_group = raw_file_split(mzmlFile_name, mzmlFile_path, scan_id, window_length=100)
        #filter_chunks(scan_id, peaks_group, label_file, output_file,output_label_0_file)
        
        output_label_0_filter_file_name = output_label_0_filter_file +str(scan_id)+'.csv'
        extract_label0_chunks(scan_id, peaks_group, falsh_file,output_label_0_filter_file_name)




def extract_label0_chunks(scan_id, peaks_group, falsh_file,output_label_0_file):
    df = pd.read_csv(falsh_file, sep='\t')
    labels = df[(df['ScanNum'] ==scan_id)]
    chunks_0 = []
    
    for chunk in peaks_group:
        contains_range = False

        # Iterate over the labels
        for index, row in labels.iterrows():
            start_mz = row['RepresentativeMzStart']
            end_mz = row['RepresentativeMzEnd']

            # Check if any m/z value in the chunk falls within the range
            if any(start_mz <= mz <= end_mz for mz, _ in chunk):
                contains_range = True
                break

        # If the chunk contains the desired m/z range, add it to the output list
        if not contains_range:
            chunk_label_0 = chunk + [0]
            chunks_0.append(chunk_label_0 )

    df_0 = pd.DataFrame(chunks_0)
    df_0.to_csv(output_label_0_file, mode='a',header=False, index=False)






def traindata_drop_duplicates(traindata_file_path, output_file_path):
    df = pd.read_csv(traindata_file_path)
    df = df.drop_duplicates()
    df.to_csv(output_file_path, index=False)
    
'''
traindata_drop_duplicates(traindata_file_path = "D:/gitclone/denoise/data/train/train_ms2_data3_4_5_6_7.csv", 
                          output_file_path =  "D:/gitclone/denoise/data/train/train_ms2_data3_4_5_6_7_dropduplicates.csv")
'''





#mzmlFile_name='20230709S4cytcMSeCE70rd01_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/1'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=119,end_scan_num=148)  

#mzmlFile_name='202307215mixproteinMSeCV70105V_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/2'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=88,end_scan_num=140)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=88,end_scan_num=140)

#mzmlFile_name='20230803NANO3300asynuclinCV70V_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/3'           
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=25,end_scan_num=38)

#mzmlFile_name='20231013STANDARDESIMSECV70VBETAruqiumixcytCS1_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/5'           
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)

#mzmlFile_name='20231013STANDARDESIMSECV70VBETAruqiumixlysS1_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/4'           
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)


#mzmlFile_name='20240104onlineapomyoglobin10ulmin3point5kvS3_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/6'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=44)  

#mzmlFile_name='20240104onlinehemomyoglobin10ulmin3point5kvS1_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/7'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=45) 
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=45)



#mzmlFile_name='20240911LAOC30015IKV60V2_func01.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/8'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=1,end_scan_num=28) 
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=1,end_scan_num=28)


#mzmlFile_name='20240911MBC30015IKV60V3_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/9'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=10,end_scan_num=20) 
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=10,end_scan_num=20)


#mzmlFile_name='20230727RPLCribosomalproteinCV2070Vscantime5sS1_func1.mzML'
#mzmlFile_path='D:/gitclone/denoise/data/mzmlfile/13'  
#Flashdeconv_filter(mzmlFile_name, mzmlFile_path,start_scan_num=1,end_scan_num=320)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=1,end_scan_num=320)







#mzmlFile_name = "20230803NANO3300asynuclinCV70V_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_3/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=20,end_scan_num=39)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=20,end_scan_num=39)



mzmlFile_name = "20230709S4cytcMSeCE70rd01_func2.mzML"
mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_1/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=119,end_scan_num=146)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=119,end_scan_num=146)




#mzmlFile_name = "20231013STANDARDESIMSECV70VBETAruqiumixlysS1_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_4/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)



#mzmlFile_name = "20231013STANDARDESIMSECV70VBETAruqiumixcytCS1_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_5/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=13,end_scan_num=20)


#mzmlFile_name = "20240104onlineapomyoglobin10ulmin3point5kvS3_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_6/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=44)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=44)


#mzmlFile_name = "20240104onlinehemomyoglobin10ulmin3point5kvS1_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_7/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=45)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=32,end_scan_num=45)


#mzmlFile_name = "20230727RPLCribosomalproteinCV2070Vscantime5sS1_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_13/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=1,end_scan_num=320)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=1,end_scan_num=320)



#mzmlFile_name = "202307215mixproteinMSeCV70105V_func2.mzML"
#mzmlFile_path = "D:/gitclone/denoise/data/mzmlfile/FUNC2_2/"
#Flashdeconv_filter_func2(mzmlFile_name, mzmlFile_path,start_scan_num=88,end_scan_num=140)
#topFD_flashdeconv_union(mzmlFile_name, mzmlFile_path,start_scan_num=88,end_scan_num=140)

#手动检查信号
#label_1(mzmlFile_name, mzmlFile_path)
#提取chunks
extract_label1_chunks(mzmlFile_name, mzmlFile_path)




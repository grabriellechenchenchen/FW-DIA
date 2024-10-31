# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:51:46 2024

@author: 555
"""

import sys
sys.path.append("D:/Software/topdown/Unidec604/diaSC")  
from ms2 import *
import pandas as pd
import os
import time
from multiprocessing import Pool
import ast
from untitled11_singleProcess import *
from ast import literal_eval
import collections
import csv
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
                                       
def ms2_eic_update(ms2_file_path, ms2_mzmlFile_name,resolution, rt_tolerance):
    
    ms2js_df = pd.read_csv(ms2_file_path+'/topfd/MS2/ms2_data_filter.csv')
    mzml_file = ms2_file_path+'/'+ms2_mzmlFile_name
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file,exp)
    ms2js_df = ms2js_df.astype({
        "Scan":float,
        "Retention Time":	float,
        'ID': float, 
        'Mono Mass': float, 
        'Charge': float
        })
    
    all_time_intensities=[]
    f_intensity_list = [] 
    i=0
    ms2js_df =  ms2js_df.head(10) 
    for  _, rt_row in ms2js_df.iterrows():
        i+=1
        print(i)
        f_rt = float(rt_row['Retention Time'])
        f_scan = float(rt_row['Scan'])
        f_deconv_ID = int(float(rt_row['ID']))
        f_mz = eval(rt_row['MZ Intensity Pairs'])
        f_mono_Mass = float(rt_row['Mono Mass'])
        f_charge = float(rt_row['Charge'])

        f_intensity = sum(i[1] for i in f_mz)
        f_intensity_list.append(f_intensity)
        
        if len(f_mz) < 2:
            mz_interval = Interval(f_mz[0][0] - resolution, f_mz[0][0] + resolution) if f_mz else Interval(0, 0)
        elif (f_mz[1][0] - f_mz[0][0])>0.7:   
            mz_interval = Interval(f_mz[0][0]-resolution,f_mz[0][0]+resolution)
        else:
            mz_interval = Interval(f_mz[0][0],f_mz[-1][0])

        time_dependent_intensities = []
        for spectrum in exp:
            rt = spectrum.getRT()
            if (f_rt - rt_tolerance) <= rt <= (f_rt + rt_tolerance):
                intensity_sum = np.sum([peak.getIntensity() for peak in spectrum if peak.getMZ() in mz_interval])
                time_dependent_intensities.append((rt, intensity_sum))
            else:
                time_dependent_intensities.append((rt, 0))  
    
        values = np.array(time_dependent_intensities)
        max_intensity = np.max(values[:, 1])
        scaling_factor = max_intensity / 100 if max_intensity > 0 else 1
        normalized_intensities = [(rt, intensity/scaling_factor) for rt, intensity in time_dependent_intensities]
        all_time_intensities.append(normalized_intensities)

    rt_columns = []
    intensity_values = []
    
    for intensities in all_time_intensities:
        rt_intensity_dict = {rt: intensity for rt, intensity in intensities}
        rt_columns.append(rt_intensity_dict)

    rt_df = pd.DataFrame(rt_columns) 
    ms2js_df['f_intesnity'] = f_intensity_list
    ms2js_df_with_eic = pd.concat([ms2js_df, rt_df], axis=1)
    new_csv_file_path = ms2_file_path + '/topfd/MS2/ms2_data.csv'
    ms2js_df_with_eic.to_csv(new_csv_file_path, index=False)



def filter_ms2candidate(ms1_p_df, ms2_df, precursor_id):
    
    timeTolerance_second = 40   

    p_row =  ms1_p_df.loc[ms1_p_df['id'] ==precursor_id]
    p_mass = float(p_row['Mono Mass'])
    p_charge = eval(p_row['Charge'].iloc[0]) 
    p_charge_max = max(p_charge)
    p_curve = eval(p_row['gaussian curve'].iloc[0])
    ini_candidate_df = ms2_df.loc[(ms2_df['Retention Time'] > (p_curve[0][0]-timeTolerance_second )) & (ms2_df['Retention Time'] <(p_curve[-1][0]+timeTolerance_second )) & (ms2_df['Mono Mass'] < (p_mass-20)) & (ms2_df['Charge'] < p_charge_max),:]
   
    
    return ini_candidate_df
    




def eic_behaviour_calculate(p_curve_df, f_curve_df):
    
    dfA = p_curve_df.iloc[:, 7:]
    dfB = f_curve_df.iloc[:, 7:]

    B = dfB.values  
    A = dfA.values  
    
    
    A_mean = A.mean()
    A_std = A.std()
    A_normalized = (A - A_mean) / A_std  
    
    
    B_mean = B.mean(axis=1, keepdims=True)  
    B_std = B.std(axis=1, ddof=0, keepdims=True)  
    B_normalized = (B - B_mean) / B_std  
    
    correlations = np.dot(B_normalized, A_normalized) / (B_normalized.shape[1])
    

    df['Pearson_Correlation'] = correlations
    print(df[['Pearson_Correlation']].head())
    
    df_sorted = df.sort_values(by='Pearson_Correlation', ascending=False)
    print(df_sorted[['Pearson_Correlation']].head())
    
    threshold = 0.8
    df_similar = df[df['Pearson_Correlation'] > threshold]
    print(df_similar[['Pearson_Correlation']].head())

    writer.writerow(['precursor id','precursor mono mass','precursor charge','fragment id','fragment mono mass','fragment ms2 scan','fragment ms2 rt','fragment deconv id','fragment charge','fragment intensity','fragment mz internsity pairs'])

    return result_df
    

def draw_p_info(p_curve):
    plt.figure(figsize=(10, 6))
    x_data = [i[0] for i in p_curve]
    y_data = [i[1] for i in p_curve]
    plt.plot(x_data, y_data, 'bo',label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
 
def calculate_pearson_correlation(A_trimmed, B_trimmed):
    A_intensities = [intensity for _, intensity in A_trimmed]
    B_intensities = B_trimmed
    
    if len(A_intensities) < 2 or len(B_intensities) < 2:
        return None
    
    corr, _ = pearsonr(A_intensities, B_intensities)
    return corr


from interval import Interval
def align_and_trim_B_curve(A_curve, B_curve, B_retention_time, A_rt_dict, B_rt_dict):
    rt_tolerance = 100
    A_times = [time for time, _ in A_curve]
    A_idx = [idx for rt, idx in A_rt_dict.items() if (A_times[0]-2) <= float(rt) <= (A_times[-1]+2)]
   
    
    B_times = []
    B_times.append(B_retention_time -  rt_tolerance)
    B_times.append(B_retention_time +  rt_tolerance)
    B_idx = [idx for rt, idx in B_rt_dict.items() if (B_times[0]-2) <= float(rt) <= (B_times[-1]+2)]
    
    
   
    start_idx = max(A_idx[0], B_idx[0])
    end_idx = min(A_idx[-1], B_idx[-1])
    
    if start_idx > end_idx:
        return None,None
    
    
   
    A_trimmed_rt = [float(k) for k, v in A_rt_dict.items() if v in Interval(start_idx,end_idx) ]
    A_trimmed_rt.sort()
    A_trimmed = [(time, intensity) for time, intensity in A_curve if (A_trimmed_rt[0]-2) <= time <= (A_trimmed_rt[-1]+2)]
    
    B_trimmed = [B_curve[i] for i in range(start_idx,end_idx+1)]
    
    return A_trimmed, B_trimmed


def calculate_pearson_matrix(A_curves, B_curves, B_retention_times, A_rt_dict, B_rt_dict):
    
    A_matrix = np.array(A_curves)
    #print("A_matrix:", A_matrix)
    B_matrix = np.array(B_curves)
    #print("B_matrix.shape: ",B_matrix.shape)  # m x 354
    pearson_matrix = []
    p_value_matrix = []
    for i, A_curve in enumerate(A_matrix):
        
        row_corrs = []
        row_p_value = []
        for j, B_curve in enumerate(B_matrix):
            #print("j:",j)
            B_retention_time = B_retention_times[j]
            A_trimmed, B_trimmed = align_and_trim_B_curve(A_curve, B_curve, B_retention_time, A_rt_dict, B_rt_dict)
            
            if A_trimmed is None or B_trimmed is None:
                row_corrs.append(None)
                row_p_value.append(None)
                continue
            A_intensities = [intensity for _, intensity in A_trimmed]
            B_intensities = B_trimmed
            
            if len(A_intensities) < 2 or len(B_intensities) < 2:
                row_corrs.append(None)
                row_p_value.append(None)
                continue
            
            corr, p_value = pearsonr(A_intensities, B_intensities)
            row_corrs.append(corr)
            row_p_value.append(p_value)

        pearson_matrix.append(row_corrs)
        p_value_matrix.append(row_p_value)
    
    return np.array(pearson_matrix), np.array(p_value_matrix)


import pyopenms
def RT_dict(mzml_file):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file,exp)
    rt_dict = {}
    count = 0
    for i in exp:
        rt = str(i.getRT())
        rt_dict[rt] = count
        count +=1
    return rt_dict
    
    
def fd_see_speedup(ms1_file_path,ms1_mzml_file, ms2_mzml_file):
    
    ms1_p_info = ms1_file_path +'/MS1_data_new_gaussian.csv'
    ms1_p_df = pd.read_csv(ms1_p_info)
    
    ms2_df = pd.read_csv(ms1_file_path +'/topfd/MS2/eic_ms2_data_final.csv')
    ms2_df = ms2_df.astype({
        'Scan': float,
        'Retention Time':float,
        'ID':float,
        'Mono Mass':float,
        'Charge':float,
        'f_intensity':float
        })

    result_folder_path = ms1_file_path+'/result/'
    pearson_result_file = result_folder_path+'pearson_calculation.csv'
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)
    
    A_rt_dict = RT_dict(ms1_mzml_file)
    B_rt_dict = RT_dict(ms2_mzml_file)
    with open(ms1_file_path + '/B_rt_dict.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(B_rt_dict, jsonfile, ensure_ascii=False, indent=4)
        
    p_id_list = [1]
    with open(pearson_result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['precursor id','precursor mono mass','precursor charge','fragment id','fragment mono mass','fragment ms2 scan','fragment ms2 rt','fragment deconv id','fragment charge','fragment intensity','fragment mz internsity pairs','corr', 'p_value'])
        for  _, p_row in ms1_p_df.iterrows():
            p_mass = float(p_row['Mono Mass'])
            p_id = float(p_row['id'])
            if p_id <3000:
                A=p_row
                p_charge = eval(p_row['Charge'])
                p_charge_max = max(p_charge)
                p_curve = eval(p_row['gaussian curve'])
                p_curve_y = [i[1] for i in p_curve]
                #draw_p_info(p_curve)
                B = filter_ms2candidate(ms1_p_df = ms1_p_df, ms2_df=ms2_df, precursor_id=p_id)
                A_curves = []
                A_curves.append(p_curve)
                B_retention_times = B['Retention Time'].tolist()
                B_curves = B.iloc[:, 7:].values
                
                pearson_matrix,p_value_matrix = calculate_pearson_matrix(A_curves, B_curves, B_retention_times, A_rt_dict, B_rt_dict)
                pearson_matrix = pearson_matrix[0]
                p_value_matrix = p_value_matrix[0]
                #print(len(pearson_matrix))
                #print(len(p_value_matrix))
                #print(B.shape)
                count = 0
                for index, row in B.iterrows():
                    writer.writerow([p_id, p_mass,p_charge_max, 
                                    count+1, row['Mono Mass'], row['Scan'], row['Retention Time'], row['ID'], row['Charge'], row['f_intensity'], row['MZ Intensity Pairs'], 
                                    pearson_matrix[count], p_value_matrix[count]])
                    count += 1


def filter_pearson_threshold(ms1_file_path, pearson_threshold):
    folder_name = "run_pearson_" +str(pearson_threshold).replace(".", "_")
    result_folder_path = ms1_file_path+'/result/'+ folder_name +"/"
    pearson_result_file = result_folder_path+'pearson_result_'+str(pearson_threshold).replace(".", "_")+'.csv'
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)
    result_df = pd.read_csv(ms1_file_path+'/result/pearson_calculation.csv')   
    result_df = result_df.astype({
        'corr':float, 
        'p_value':float,
        })
    
    pearson_threshold_df = result_df.loc[(result_df['corr'] > pearson_threshold)]
    pearson_threshold_df.to_csv(pearson_result_file, index=False)
    print(f'pearson系数大于{pearson_threshold}的结果已存在{pearson_result_file}')
    
    
    mapping_to_msalign(result_folder_path,pearson_result_file,pearson_threshold)
    print(f'pearson系数大于{pearson_threshold}的msallign已存在{pearson_result_file}')


def mapping_to_msalign(result_folder_path,pearson_result_file,pearson_threshold):
    pearson_df = pd.read_csv(pearson_result_file)  
    precursor_count = int(float(pearson_df.iloc[-1,0]))
    print("precursor_count: ", precursor_count)
    with open(result_folder_path + 'pearson_result_'+str(pearson_threshold).replace(".", "_")+'_ms2.msalign','a') as f:
        count = 0
        for i in range(1,precursor_count+1):
            group_rows = pearson_df.loc[(pearson_df['precursor id']==i),:]
            row_count = int(group_rows.shape[0])
            if row_count<5:
                print("precursor id: ", i)
                print( "rows count:",row_count )
                continue
            else:
                f.write('BEGIN IONS\n')
                f.write('ID='+str(count)+'\n')
                f.write('FRACTION_ID=0\nFILE_NAME=D:/experiment.mzML\n')
                f.write('SCANS='+str(count)+'\n')
                f.write('RETENTION_TIME='+str(count)+'\n')
                f.write('LEVEL=2\nACTIVATION=HCD\n')
                f.write('MS_ONE_ID='+str(count)+'\n')
                f.write('MS_ONE_SCAN='+str(count)+'\n')
                
                
                p_mono_mass = float(group_rows.iloc[0,1])
                p_charge = float(group_rows.iloc[0,2])
                p_mz =  (p_mono_mass+p_charge) / p_charge
                
                f.write('PRECURSOR_MZ='+str(p_mz)+'\n')
                f.write('PRECURSOR_CHARGE='+str(p_charge)+'\n')
                f.write('PRECURSOR_MASS='+str(p_mono_mass)+'\n')
                f.write('PRECURSOR_INTENSITY='+ str(10000000) +'\n')
                
                for  _, rt_row in group_rows.iterrows():
                    f.write(str(rt_row['fragment mono mass'])+'\t'+str(rt_row['fragment intensity'])+'\t'+str(rt_row['fragment charge'])+'\n')
                    #f.write(str(j)+'\t'+str(10000)+'\t'+str(1)+'\n')
                f.write('END IONS')
                f.write('\n\n')
                count=count+1



def sort_ms2_info(ms2_file_path):
    df = pd.read_csv(ms2_file_path+'/topfd/MS2/ms2_data.csv')
    df_sorted = df.sort_values(by=['Scan', 'ID'])
    df_sorted.to_csv(ms2_file_path+'/topfd/MS2/ms2_data_sorted_file.csv', index=False)
    
    file_to_delete = ms2_file_path+'/topfd/MS2/ms2_data.csv'
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
    if os.path.exists(ms2_file_path+'/topfd/MS2/ms2_data_sorted_file.csv'):
        os.rename(ms2_file_path+'/topfd/MS2/ms2_data_sorted_file.csv', ms2_file_path+'/topfd/MS2/ms2_data.csv')
        



def find_occupied_numbers_and_max_mono_mass(file_path):
    scan_tolerance = 2

    df = pd.read_csv(file_path)
    
    df['Scan'] = df['Scan'].apply(literal_eval)
    df['Charge'] = df['Charge'].apply(literal_eval)
    
    occupied_numbers = set()
    scan_to_mono_mass = {}
    scan_to_charge = {}
    
    for index, row in df.iterrows():
        scan_list = row['Scan']
        scan_list.sort()
        mono_mass = row['Mono Mass']
        charge = row['Charge']
        
        if len(scan_list) > 1:
            start, end = int(scan_list[0]), int(scan_list[-1])
            occupied_numbers.update(range(start-scan_tolerance, end + 1+scan_tolerance))
            for scan in range(start-scan_tolerance, end + 1+scan_tolerance):
                if scan not in scan_to_mono_mass:
                    scan_to_mono_mass[scan] = []
                    scan_to_charge[scan] = []
                scan_to_mono_mass[scan].append(mono_mass)
                scan_to_charge[scan].append(int(charge[-1]))
        else:
            scan = int(scan_list[0])
            occupied_numbers.update(range(scan-scan_tolerance, scan + 1+scan_tolerance))
            for scan in range(scan-scan_tolerance, scan + 1+scan_tolerance):
                if scan not in scan_to_mono_mass:
                    scan_to_mono_mass[scan] = []
                    scan_to_charge[scan] = []
                scan_to_mono_mass[scan].append(mono_mass)
                scan_to_charge[scan].append(int(charge[-1]))
    
    max_mono_mass_per_scan = {scan: max(mono_masses) for scan, mono_masses in scan_to_mono_mass.items()}
    max_mono_mass_per_scan = {k: max_mono_mass_per_scan[k] for k in sorted(max_mono_mass_per_scan.keys())}

    max_charge_per_scan = {scan: max(c) for scan, c in scan_to_charge.items()}
    max_charge_per_scan  = {k: max_charge_per_scan [k] for k in sorted(max_charge_per_scan .keys())}

    return sorted(occupied_numbers), max_mono_mass_per_scan, max_charge_per_scan



def filter_ms2_data(ms2_file_path, occupied_numbers, max_mono_mass_per_scan,max_charge_per_scan):
    ms2js_df = pd.read_csv(ms2_file_path+'/topfd/MS2/ms2_data.csv')
    ms2js_df = ms2js_df.astype({
        "Scan":float,
        "Retention Time":	float,
        'ID': float, 
        'Mono Mass': float, 
        'Charge': float
        })
    
    
    filter_scan = set(occupied_numbers)
    ms2js_filter_df = ms2js_df[ms2js_df['Scan'].isin(filter_scan)]
    print(ms2js_filter_df.shape)   
    
   
    all_filtered_dfs = [] 
    for scan, max_mono_mass in sorted(max_mono_mass_per_scan.items()):
        ms2js_filter_df = ms2js_df[(ms2js_df['Scan'] == scan) & (ms2js_df['Mono Mass'] < max_mono_mass)]
        all_filtered_dfs.append(ms2js_filter_df)

    combined_df = pd.concat(all_filtered_dfs, ignore_index=True)
    print(combined_df.shape)
    
    
    all_filtered_dfs2 = [] 
    for scan, charge in sorted(max_charge_per_scan.items()):
        ms2js_filter_df = combined_df[(combined_df['Scan'] == scan) & (combined_df['Charge'] < charge)]
        all_filtered_dfs2.append(ms2js_filter_df)

    combined_df2 = pd.concat(all_filtered_dfs2, ignore_index=True) 
    print(combined_df2.shape)
    
    new_csv_file_path = ms2_file_path + '/topfd/MS2/ms2_data_filter.csv'
    combined_df2.to_csv(new_csv_file_path, index=False)
    print(f"Processed data saved to {new_csv_file_path}")

    with open(ms2_file_path + '/topfd/MS2/max_mono_mass_per_scan.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(max_mono_mass_per_scan, jsonfile, ensure_ascii=False, indent=4)
    print("max_mono_mass_per_scan has been written")
    
    with open(ms2_file_path + '/topfd/MS2/max_charge_per_scan.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(max_charge_per_scan, jsonfile, ensure_ascii=False, indent=4)
    print("max_mono_mass_per_scan has been written ")
    
    
    
def optional_scan_limit(ms2_file_path,scan_start,scan_end):   
    ms2js_df = pd.read_csv(ms2_file_path+'/topfd/MS2/ms2_data_filter.csv')
    ms2js_df = ms2js_df.astype({
        "Scan":float
        })
    new_ms2js_df = ms2js_df[(ms2js_df['Scan'] <= scan_end ) & (ms2js_df['Scan'] >= scan_start )]
    print(f'After scan limit: {new_ms2js_df.shape[0]}')
    
    new_csv_file_path = ms2_file_path + '/topfd/MS2/ms2_data_filter_scanLimit.csv'
    new_ms2js_df.to_csv(new_csv_file_path, index=False)
    print(f"New processed data saved to {new_csv_file_path}")
    
    

    
    
start_time = time.time()

ms2_file_path =  ""
ms2_mzmlFile_name = "ms2_denoised_ms_edit_centroid.mzML"

MS2_run_topfd(ms2_file_path = ms2_file_path,ms2_mzmlFile_name = ms2_mzmlFile_name)
fd_(ms2_file_path)
sort_ms2_info(ms2_file_path)

    
file_path = "MS1_data_new.csv"
occupied_numbers, max_mono_mass_per_scan,max_charge_per_scan = find_occupied_numbers_and_max_mono_mass(file_path)
filter_ms2_data(ms2_file_path, occupied_numbers, max_mono_mass_per_scan,max_charge_per_scan)  

optional_scan_limit(ms2_file_path = ms2_file_path,scan_start = 85 ,scan_end = 150)
single_thread_processing(ms2_file_path, ms2_mzmlFile_name, 0.5, 12) 

fd_see_speedup(ms1_file_path="",
               ms1_mzml_file = "ms1_denoised_ms_centroid.mzML",
               ms2_mzml_file = "ms2_denoised_ms_edit_centroid.mzML")


pearson_threshold_list = [0.8,0.9, 0.95]
for i in pearson_threshold_list:
    filter_pearson_threshold(ms1_file_path=ms2_file_path, pearson_threshold=i)
   
end_time = time.time()
elapsed_time = end_time - start_time
print(f"run time: {elapsed_time:.2f} 秒")

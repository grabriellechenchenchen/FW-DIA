# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:15:38 2024

@author: 555
"""


import sys
sys.path.append("D:/Software/topdown/Unidec604/diaSC")  
from pyopenms import MSExperiment, MzMLFile
import os
import launcher_sc_version3_draft
from launcher_sc_version3_draft import msaccess1,on_unidec_batchMS1_deconv_isotopicOffMode,stableUniscore,unidec_isotopicOFF_maxMass
from launcher_sc_version3_draft import run_lowerSetting_flashdeconv,flashdeconv_output_process,on_unidec_batchMS1_deconv_isotopicMono
import time
import numpy as np
import pandas as pd
import shutil
from interval import Interval
import pyopenms
import csv
from intervaltree import IntervalTree
from tqdm import tqdm
import pymzml
import matplotlib.pyplot as plt
from collections import Counter
import math


def mzml_to_txt_unidec_process(mzmlFile_name,mzmlFile_path):
    exp = MSExperiment()
    mzml = MzMLFile()
    mzml.load(mzmlFile_path +'/'+ mzmlFile_name, exp)
    output_dir = mzmlFile_path+'/unidec_isotopicOFF/msaccess_temp_txt'
    os.makedirs(output_dir, exist_ok=True)
    projectName = mzmlFile_name.replace('.mzML','')
    for i,spectrum in enumerate(exp):
        mz_array, intensity_array = spectrum.get_peaks()
        #scanId = f"{mzmlFile_name.replace('.mzML', '')}_scan_{i+1}"
        scanId=i+1
        file_path = os.path.join(output_dir, f"{scanId}.txt")
        mz_array = np.asarray(mz_array)
        intensity_array = np.asarray(intensity_array)
        data_points = np.column_stack((mz_array, intensity_array))
        np.savetxt(file_path, data_points, fmt='%.6f %.6f', delimiter=' ', newline='\n')


def flash_topfd_match_items(spec_ms1_df, fd_msalign_file):

    
    msalign_data = []
    with open(fd_msalign_file, 'r') as infile:
        current_scan_info = []
        current_rt = None
        scan_active = False
    
        for line in infile:
            line = line.strip()
    
            if line.startswith('RETENTION_TIME='):
                current_rt = math.floor(float(line.split('=')[1]))
                scan_active = False  
                current_scan_info = []  
    
            elif line == 'LEVEL=1':
                scan_active = True  
    
            elif line == 'END IONS':
                if scan_active and current_rt is not None:
                    msalign_data.append((current_rt, current_scan_info))
                scan_active = False
    
            elif scan_active:
                split_line = line.split('\t')
                mass = float(split_line[0])
                charge = float(split_line[2])
                current_scan_info.append((mass, charge))
    

    msalign_dict = {}
    for rt, scan_info in msalign_data:
        if rt not in msalign_dict:
            msalign_dict[rt] = []
        msalign_dict[rt].extend(scan_info)
    
    
    flash_align_fd = []
    for _, rt_row in spec_ms1_df.iterrows():
        flash_monoMass = float(rt_row['MonoisotopicMass'])
        if flash_monoMass > 4000:  
            flash_rt_apex = math.floor(float(rt_row['ApexRetentionTime']))
            flash_featureIndex = int(float(rt_row['FeatureIndex']))
            flash_charge_min = float(rt_row['MinCharge'])
            flash_charge_max = float(rt_row['MaxCharge'])
    
            
            if flash_rt_apex in msalign_dict:
                thisScanInfo = msalign_dict[flash_rt_apex]
    
                
                for mass, charge in thisScanInfo:
                    if flash_monoMass - 1 <= mass <= flash_monoMass + 1:
                        if flash_charge_min <= charge <= flash_charge_max:
                            flash_align_fd.append(flash_featureIndex)
                            break
    
    print('flashdeconv align with topfd number (>4000Da):', len(flash_align_fd))
    return flash_align_fd


    
def flash_align_topfd_speedup(ms1_file_path,mzmlFile_name):
    flash_fd_intersec = []
    count = 0 
    #flash default running
    flashd_mass_file = ms1_file_path +'/flashdeconv_default_output/mass.tsv'
    flashd_spec_ms1_file = ms1_file_path +'/flashdeconv_default_output/specmass_ms1.tsv'
    
    #flash low setting
    flashd_mass_file = ms1_file_path +'/flashdeconv_lowerSetting_output/mass.tsv'    
    flashd_spec_ms1_file = ms1_file_path +'/flashdeconv_lowerSetting_output/specmass_ms1.tsv'
    
    fd_msalign_file = ms1_file_path + '/topfd/MS1/ms1_spectrum_file/ms1_spectrum_ms1.msalign'
    
    spec_ms1_df = pd.read_csv(flashd_mass_file, sep='\t')
    spec_ms1_df = spec_ms1_df.astype({
        'MonoisotopicMass': float,
        'MinCharge': float,
        'MaxCharge': float,
        'StartRetentionTime': float,
        'EndRetentionTime': float,
        'ApexRetentionTime': float,
        'FeatureIndex': float
    })
    
    
    flash_align_fd = flash_topfd_match_items(spec_ms1_df, fd_msalign_file)
    
    ms1_p_info = ms1_file_path +'/MS1_data.csv'
    ms1_p_df = pd.read_csv(ms1_p_info)
    rowCount = len(ms1_p_df)
    already_store =[]
    
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(ms1_file_path + '/'+mzmlFile_name,exp)
    spec_count = 0
    scan_rt_dic = {}
    for spectrum in exp:
        spec_count = spec_count + 1
        rt_time_int = math.floor(spectrum.getRT())
        scan_rt_dic[rt_time_int-1] =   spec_count 
        scan_rt_dic[rt_time_int] =   spec_count    
        scan_rt_dic[rt_time_int+1] =   spec_count 
 
    spec_ms1ms1_df = pd.read_csv(flashd_spec_ms1_file,sep='\t')
    spec_ms1ms1_df['MonoisotopicMass_int'] = spec_ms1ms1_df['MonoisotopicMass'].apply(lambda x: math.floor(x))
    spec_ms1ms1_df['RetentionTime_int'] = spec_ms1ms1_df['RetentionTime'].apply(lambda x: math.floor(x))
    
    with open(ms1_p_info, 'a', newline='') as file:
        writer = csv.writer(file)
        
        count = rowCount
        for i in tqdm(flash_align_fd):
            
            flash_candidate_df = spec_ms1_df.loc[(spec_ms1_df['FeatureIndex']==i),['MonoisotopicMass','ApexRetentionTime','EndRetentionTime','MinCharge','MaxCharge']]
            
            mono_mass = math.floor(flash_candidate_df.iloc[0,0])
            mono_mass = flash_candidate_df.iloc[0,0]
            rt_apex = math.floor(flash_candidate_df.iloc[0,1])
            rt_end = math.floor(flash_candidate_df.iloc[0,2])
            rt_apex_scan = scan_rt_dic[rt_apex]
            rt_end_scan = scan_rt_dic[rt_end]
            feature_min_charge = float(flash_candidate_df.iloc[0,-2])
            feature_max_charge = float(flash_candidate_df.iloc[0,-1])
            
            need_info = spec_ms1ms1_df.loc[(mono_mass+1>spec_ms1ms1_df['MonoisotopicMass']) & (mono_mass-1<spec_ms1ms1_df['MonoisotopicMass']) & ((spec_ms1ms1_df['RetentionTime_int']==rt_apex) | (spec_ms1ms1_df['RetentionTime_int']==rt_apex-1) | (spec_ms1ms1_df['RetentionTime_int']==rt_apex+1)),['MonoisotopicMass','SumIntensity','MinCharge','MaxCharge','PeakMZs']]      
            
            if len(need_info) > 1:
                max_i = 0
                thisOne = 0
                for i in range(0,len(need_info)):
                    i_itensity = need_info.iloc[i,1]
                    if i_itensity>max_i:
                        thisOne = i
                        max_i = i_itensity

            else:
                thisOne = 0
  
            store_mass = need_info.iloc[thisOne,0]
   
             
            check_current_ms1 = ms1_p_df.loc[(ms1_p_df['Mono Mass']>store_mass-5) & (ms1_p_df['Mono Mass']<store_mass+5),['id']]
            if len(check_current_ms1)>0:  
                already_store.append(check_current_ms1.iloc[0,0])
                
            else:    
                store_scan = [ rt_apex_scan, rt_end_scan]  
                store_rt = [flash_candidate_df.iloc[0, 1], flash_candidate_df.iloc[0, 2]]  
                min_charge = min(need_info.iloc[thisOne, 2], feature_min_charge)
                max_charge = max(need_info.iloc[thisOne, 3], feature_max_charge)
                store_charge = [min_charge, max_charge]
                
                store_mz = str(need_info.iloc[thisOne,4] )
                data =  [float(num) for num in store_mz.split()]

                differences = [data[i+1] - data[i] for i in range(len(data) - 1)]

                threshold = 10
                groups = []
                current_group = [data[0]]
                
                for i in range(1, len(data)):
                    if data[i] - data[i - 1] > threshold:
                        groups.append(current_group)
                        current_group = [data[i]]
                    else:
                        current_group.append(data[i])
                
                groups.append(current_group)
                
                store_mz = []
                
                for k in groups:
                    store_mz.append(np.mean(k))

                if store_charge[1]-store_charge[0] != 0: 
                    if len(store_mz) >1:  
                        count=count +1
                        writer.writerow([count,'flash_fd',store_mass,store_scan, store_rt, store_charge, store_mz])

    
    ms1_p_info_2 = ms1_file_path +'/MS1_data.csv'
    ms1_p_df_2 = pd.read_csv(ms1_p_info_2)
    ms1_p_df_2.loc[ms1_p_df_2['id'].isin(already_store), 'source'] += '_fd'
    ms1_p_df_2.to_csv(ms1_file_path +'/MS1_data_new.csv', index=False)




            
            
def EIC_speedup(ms1_file_path,mzmlFile_name):
    mzml_file = ms1_file_path + '/' +mzmlFile_name
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file, exp)
    
    ms1_p_info = ms1_file_path +'/MS1_data_new.csv'
    ms1_p_df = pd.read_csv(ms1_p_info)
    scan_tolerance =4
    count = 0
    
    questionable_id  = [] 
    multiple_id = []
    too_low =[]
    store_g_curve = []
    all_rt = [spectrum.getRT() for spectrum in exp]
    all_mz = [np.array([peak.getMZ() for peak in spectrum]) for spectrum in exp]
    all_intensity = [np.array([peak.getIntensity() for peak in spectrum]) for spectrum in exp]

    for  _, rt_row in ms1_p_df.iterrows():
        monoMass = float(rt_row['Mono Mass'])
        m_id = float(rt_row['id'])
        if (m_id <10000):   
            
            print("m_id:", m_id)
            scan = eval(rt_row['Scan'])
            rtrt = eval(rt_row['Retention Time'])
            rtrt.sort()
            
            massToCharge = eval(rt_row['MZ'])
            pick_index = len(massToCharge)
            minScan = min(scan) - scan_tolerance
            maxScan = max(scan) + scan_tolerance
    
            mass_curves_x = []
            mass_curves_y = []
            all_points = []

            for i in range(0,pick_index):
                window_tolerance = massToCharge[i] / 10000
                time_dependent_intensities1 = []
                
                for spec_idx, (rt, mz, intensity) in enumerate(zip(all_rt, all_mz, all_intensity)):
                    if minScan <= spec_idx + 1 <= maxScan:
                        mask = (mz >= massToCharge[i] - window_tolerance * 2) & (mz <= massToCharge[i] + window_tolerance * 2)
                        intensity_sum = np.sum(intensity[mask])
                        time_dependent_intensities1.append([rt, intensity_sum])
                        mass_curves_x.append(rt)
                        mass_curves_y.append(intensity_sum)

                all_points.append(time_dependent_intensities1)  

                values = time_dependent_intensities1
                scan_val = [x[0] for x in values]
                intensity_val = [x[1] for x in values]
            

            max_intensity = max(mass_curves_y)
            scale_indicator = max_intensity / 100
            for oneMZ in all_points:
                for point in oneMZ:
                    point[1] = point[1]/scale_indicator


            peak_rt = []
            curve_peak_dict = {}   
            for i in all_points:  
                i_index = all_points.index(i)
                get_peaks_for_each_mz = get_peak_number_update(time_dependent_intensities = i,height=1)
                if len(get_peaks_for_each_mz) != 0:
                    peak_rt.append(get_peaks_for_each_mz)   
                curve_peak_dict[i_index] = get_peaks_for_each_mz 
                    
            
            
            if len(peak_rt) ==0:
                curve_peak_dict = {}
                for i in all_points: 
                    i_index = all_points.index(i)
                    get_peaks_for_each_mz = get_peak_number_update(time_dependent_intensities = i,height = 0)
                    if len(get_peaks_for_each_mz) !=0:
                        peak_rt.append(get_peaks_for_each_mz)  
                    curve_peak_dict[i_index] = get_peaks_for_each_mz 
                        
                        
            common_peak = common_peaks_speedup(peak_rt) 
            curve_to_delete=[]
            for curve, peak in curve_peak_dict.items():
                containCommonPeak = False
                if len(peak) == 0:
                    curve_to_delete.append(curve)
                for cp in common_peak:
                    if cp in peak:
                        containCommonPeak = True
                if not containCommonPeak:
                    curve_to_delete.append(curve)
            if len(curve_to_delete) != 0:
                mass_curves_x = []
                mass_curves_y = []
                all_points = []
                for i in range(0,pick_index):
                    if i not in curve_to_delete:
                        window_tolerance = massToCharge[i] / 10000
                        time_dependent_intensities1 = []
                        
                        for spec_idx, (rt, mz, intensity) in enumerate(zip(all_rt, all_mz, all_intensity)):
                            if minScan <= spec_idx + 1 <= maxScan:
                                mask = (mz >= massToCharge[i] - window_tolerance * 2) & (mz <= massToCharge[i] + window_tolerance * 2)
                                intensity_sum = np.sum(intensity[mask])
                                time_dependent_intensities1.append([rt, intensity_sum])
                                mass_curves_x.append(rt)
                                mass_curves_y.append(intensity_sum)
    
                        all_points.append(time_dependent_intensities1) 
    
                        values = time_dependent_intensities1
                        scan_val = [x[0] for x in values]
                        intensity_val = [x[1] for x in values]

                
                
                max_intensity = max(mass_curves_y)
                scale_indicator = max_intensity / 100
                for oneMZ in all_points:
                    for point in oneMZ:
                        point[1] = point[1]/scale_indicator
             
                peak_rt = []
                for i in all_points:  
                    get_peaks_for_each_mz = get_peak_number_update(time_dependent_intensities = i,height=1)
                    if len(get_peaks_for_each_mz) != 0:
                        peak_rt.append(get_peaks_for_each_mz)   
                    

                if len(peak_rt) ==0:
                    for i in all_points: 
                        get_peaks_for_each_mz = get_peak_number_update(time_dependent_intensities = i,height = 0)
                        if len(get_peaks_for_each_mz) !=0:
                            peak_rt.append(get_peaks_for_each_mz)  
                       

            info = []
            p0 = []
            common_peaks_x= []
            for i in common_peak:
                rt = mass_curves_x[i]
                rt_intensity = 0
                count = 0
                for j in all_points:
                    for k in j:
                        if (k[0] == rt )& (k[1]!=0):
                            rt_intensity =rt_intensity +k[1]
                            count = count +1
                
                gaussian_rt_intensity = rt_intensity / count
                info.append((gaussian_rt_intensity, rt, 15))
                p0.extend([gaussian_rt_intensity, rt, 15])
                common_peaks_x.append(rt)
            

            max_gaussian_intensity = np.max([x[0] for x in info])


            mass_curves_y = mass_curves_y / scale_indicator
            mass_curves_y[mass_curves_y > 2 * max_gaussian_intensity] = 0
            

            gaussian_points = np.array([(x, y) for x, y in zip(mass_curves_x, mass_curves_y)])
            gaussian_points = np.unique(gaussian_points, axis=0)
            sorted_indices = np.argsort(gaussian_points[:, 0])
            gaussian_points = gaussian_points[sorted_indices]

            mass_curves_x, mass_curves_y = gaussian_points[:, 0], gaussian_points[:, 1]

            y_fit,popt = gaussian_curve2_update(gaussian_curve_number = len(common_peak), p0 =p0, x_data=mass_curves_x, y_data=mass_curves_y, common_peaks = info)
            
            store_gaussian = []
            for i in range(0,len(mass_curves_x)):
                store_gaussian.append((mass_curves_x[i],y_fit[i]))
            
            store_gaussian = list(set(store_gaussian))
            store_gaussian.sort()
            store_g_curve.append(store_gaussian)
            
            if len(popt) <4:
                height = popt[0]
                if height < 1:
                    too_low.append(m_id)
                gaussian_rt = popt[1]
                if gaussian_rt  not in Interval(rtrt[0]-20,rtrt[-1]+20):
                    if gaussian_rt  not in Interval(info[0][1]-20,info[0][1]+20):
                        questionable_id.append(m_id)
                        print("--------------------warn 1--------------")

            else:
                multiple_id.append(m_id)
                print('----------warn 2--------------')
    
    
    print('too_low_id:',too_low)
    print('length of too_low_id:',len(too_low))        
    print('questionable_id:',questionable_id)
    print('length of questionable_id:',len(questionable_id))
    print('multiple_id:',multiple_id)
    print('length of multiple_id:',len(multiple_id))

    ms1_p_info_2 = ms1_file_path +'/MS1_data_new.csv'
    ms1_p_df_2 = pd.read_csv(ms1_p_info_2)

    ms1_p_df_2 = ms1_p_df_2.loc[(ms1_p_df_2['id']<10000),:]
    ms1_p_df_2['gaussian curve'] = store_g_curve
    ms1_p_df_2.to_csv(ms1_file_path +'/MS1_data_new_gaussian.csv', index=False) 

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm


def gaussian(x, amplitude, mean, stddev):
    return amplitude * norm.pdf(x, mean, stddev)

def update_gaussin_finding(common_peaks_list,curves,window_size=40):
    for peak in common_peaks_list:
        x_peak, h_peak = peak
        fitted_peaks = []
        
        for curve_x, curve_y in curves:
            peak_indices = np.abs(curve_x - x_peak) < window_size
            peak_x = curve_x[peak_indices]
            peak_y = curve_y[peak_indices]
            if len(peak_x) > 5:
                p0 = [h_peak, x_peak, 15.0]  
                popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=p0)
                fitted_peaks.append((popt[0], popt[1], popt[2]))

        avg_amplitude = np.mean([p[0] for p in fitted_peaks])
        avg_mean = np.mean([p[1] for p in fitted_peaks])
        avg_stddev = np.mean([p[2] for p in fitted_peaks])
        
        print(f"Peak at x={x_peak}: Avg Amplitude={avg_amplitude}, Avg Mean={avg_mean}, Avg Stddev={avg_stddev}")
        

def common_peaks_speedup(peak_rt):

    seen = Counter()
    for rt_list in peak_rt:
        seen.update(rt_list)
    
    threshold = math.floor(len(peak_rt) / 3)
    common_peak = [rt for rt, count in seen.items() if count > threshold]
    if len(common_peak) == 1:
        return common_peak
    elif len(common_peak) == 0:
        threshold = math.floor(len(peak_rt) / 4)
        common_peak = [rt for rt, count in seen.items() if count > threshold]
    
    common_peak.sort()
    report_peak = []
    n = len(common_peak)
    for p in range(n - 1):
        if common_peak[p + 1] - common_peak[p] == 1:
            if seen[common_peak[p + 1]] > seen[common_peak[p]]:
                report_peak.append(common_peak[p + 1])
            else:
                report_peak.append(common_peak[p])
        else:
            report_peak.append(common_peak[p])

    if common_peak[-1] - common_peak[-2] != 1:
        report_peak.append(common_peak[-1])
    
    report_peak = list(set(report_peak))
    return report_peak





def calculate_bounds(common_peaks, delta, max_width):
    lower_bounds = []
    upper_bounds = []
    
    for i, (height, cen,width) in enumerate(common_peaks):
        
        lower_bounds.append(0)
        upper_bounds.append(np.inf)
        
        lower_bounds.append(cen - delta)
        upper_bounds.append(cen + delta)
        
        lower_bounds.append(0)
        upper_bounds.append(max_width)
    
    bounds = (lower_bounds, upper_bounds)
    
    return bounds



def calculate_weights(x, common_peaks, scale=1):
    
    common_peaks_x = [x[1] for x in common_peaks]
    
    weights = np.zeros_like(x)
    for peak in common_peaks_x:
        
        weights += np.exp(-((x - peak)**2) / (2 * (scale**2)))
   
    weights = np.maximum(weights, 0.1)
    
    return weights


def estimate_initial_width(x_data, y_data, common_peaks):
    widths = []
    for i in common_peaks:
        peak_x = i[1]
        idx = np.argmin(np.abs(x_data - peak_x))
        local_y = y_data[max(0, idx-3):min(len(y_data), idx+3)]
        width = np.std(local_y)
        widths.append(width)
    return widths

from scipy.optimize import curve_fit
def gaussian_curve2_update(gaussian_curve_number, p0,x_data, y_data, common_peaks):
    weights = calculate_weights(x_data, common_peaks, scale=1)
    bounds = calculate_bounds(common_peaks = common_peaks, delta = 5, max_width = 50.0)
    
    popt, pcov = curve_fit(multi_gaussian_update, x_data, y_data, p0=p0,maxfev=500000, sigma = 1/weights, bounds = bounds)  
    
    y_fit = multi_gaussian_update(x_data, *popt)

    return y_fit,popt


def multi_gaussian_update(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    return y




from scipy.signal import find_peaks
def get_peak_number_update(time_dependent_intensities,height):
    x = [x[0] for x in time_dependent_intensities]
    x = np.array(x)
    y = [x[1] for x in time_dependent_intensities]
    y = np.array(y)
    peaks, _ = find_peaks(y,height = height)
    
    
    
    return list(peaks)   




 
def deconvolution_start(ms1_file_path,mzmlFile_name):
    start_time = time.time()
    
    taskcode1 =  launcher_sc_version3_draft.run_default_flashdeconv(ms1_file_path,mzmlFile_name)  
    
    mzml_to_txt_unidec_process(mzmlFile_name,ms1_file_path)
    
    on_unidec_batchMS1_deconv_isotopicOffMode(file_folder= ms1_file_path +'/unidec_isotopicOFF/msaccess_temp_txt') 
    
    stableUniscore(file_folder = ms1_file_path +'/unidec_isotopicOFF/msaccess_temp_txt/',mzmlFile_path = ms1_file_path +'/unidec_isotopicOFF/',mono=False)
    unidec_maxMass = unidec_isotopicOFF_maxMass(ms1_file_path = ms1_file_path+'/unidec_isotopicOFF')   

    
    default_flashdeconv_massfile = ms1_file_path + '/flashdeconv_default_output/mass.tsv'
    df1 = pd.read_csv(default_flashdeconv_massfile, sep='\t')
    df1['MonoisotopicMass'] = df1['MonoisotopicMass'].astype(float)
    max_featureMass = df1['MonoisotopicMass'].max()

    
    #wugedanbai run 1022: max_featureMass=37021.6966  unidec_maxMass = 165075.0
    if max_featureMass > unidec_maxMass:
        flashdeconv_output_process() 
    
    else:  
        
        run_lowerSetting_flashdeconv(ms1_file_path,mzmlFile_name)  
        
        if not os.path.exists(ms1_file_path+'/unidec_isotopicMono'):
            os.mkdir(ms1_file_path+'/unidec_isotopicMono')
            
        if not os.path.exists(ms1_file_path+'/unidec_isotopicMono/'+'msaccess_temp_txt'):    
            os.mkdir(ms1_file_path+'/unidec_isotopicMono/'+'msaccess_temp_txt')
            isotopicOFF_txt_path = ms1_file_path +'/unidec_isotopicOFF/msaccess_temp_txt/'
            for file in os.listdir(isotopicOFF_txt_path):
                if '_unidecfiles' not in file:
                    shutil.copyfile(ms1_file_path +'/unidec_isotopicOFF/msaccess_temp_txt/'+file, ms1_file_path +'/unidec_isotopicMono/msaccess_temp_txt/'+file) 
        on_unidec_batchMS1_deconv_isotopicMono(file_folder= ms1_file_path +'/unidec_isotopicMono/msaccess_temp_txt')
        
        
        stableUniscore(file_folder = ms1_file_path +'/unidec_isotopicMono/msaccess_temp_txt/',mzmlFile_path = ms1_file_path +'/unidec_isotopicMono/',mono=True)
        
        launcher_sc_version3_draft.unidec_isotopicMono_maxMass(ms1_file_path = ms1_file_path+'/unidec_isotopicMono')
        
        launcher_sc_version3_draft.mapping_uni_mono_to_off(ms1_file_path)  
        launcher_sc_version3_draft.uni_mono_align_result_combine(ms1_file_path)    
        launcher_sc_version3_draft.uni_off_result_combine(ms1_file_path)  
        launcher_sc_version3_draft.uni_off_result(ms1_file_path)    

        launcher_sc_version3_draft.unidec_mono_intersec_flash(ms1_file_path,mzmlFile_name)   
        launcher_sc_version3_draft.uni_off_align_flash(ms1_file_path,mzmlFile_name) 
        #TODO: run topFD in this step and stor the result in ms1_file_path +'/topfd/MS1'
        flash_align_topfd_speedup(ms1_file_path,mzmlFile_name)

        EIC_speedup(ms1_file_path,mzmlFile_name)
        
            
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"run time: {elapsed_time:.2f} s")


deconvolution_start(ms1_file_path = "xxx/xxx",
                    mzmlFile_name = "xxx.mzML")



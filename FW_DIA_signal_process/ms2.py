# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:00:35 2024

@author: 555
"""



import os
import subprocess
import wx
import wx.html
import multiprocessing
import wx.py as py
import os
import locale
locale.setlocale(locale.LC_ALL, 'C')
import tools.msaccess as msaccess
import subprocess
import os
import sys
import shutil
from pymzml.run import Reader
import pandas as pd
import csv
import json
sys.path.append("D:\\Software\\topdown\\Unidec604\\UniDec-v.6.0.4")

import unidec.engine as UnidecEng
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.offline as of1
import numpy as np
from unidec import tools as ud
import scipy.ndimage.filters as filt
from interval import Interval
import pymzml
import matplotlib.pyplot as plts
from sklearn.cluster import AgglomerativeClustering,DBSCAN
import matplotlib.pyplot as plt
import itertools
        



def MS2_run_lowerSetting_flashdeconv(ms2_file_path,ms2_mzmlFile_name):   
    lowerSetting_flashdecnvOutput_path = ms2_file_path + '/MS2_flashdeconv_lowerSetting_output/'
    if not os.path.exists(lowerSetting_flashdecnvOutput_path):
        os.mkdir(lowerSetting_flashdecnvOutput_path)
    
    MS2_mzml= ms2_file_path+'/'+ms2_mzmlFile_name
    
    flashdeconv_lowerSetting_running_cmd = 'FLASHDeconv -in {} -out ms2mass.tsv -out_spec ms2specmass_ms1.tsv -out_topFD ms2specmass_ms1.msalign -write_detail 1 -Algorithm:tol 20.0 20.0 10.0 -Algorithm:min_isotope_cosine 0.7 0.7 0.85 -FeatureTracing:min_trace_length 5 -merging_method 1'.format(MS2_mzml)
    ret = subprocess.run(flashdeconv_lowerSetting_running_cmd, cwd = lowerSetting_flashdecnvOutput_path)
    if ret.returncode == 0:
        print('MS2 flashdeconv running done')


def MS2_run_topfd(ms2_file_path,ms2_mzmlFile_name):   
    MS2_mzml= ms2_file_path+'/topfd/MS2/ms2_spectrum.mzML'
    topfd_ms2_running_cmd = 'D:/Software/TopPic/toppic-windows-1.6.2/topfd -r 1 {}'.format(MS2_mzml)
    ret = subprocess.run(topfd_ms2_running_cmd, cwd = ms2_file_path+'/topfd/MS2/')
    print()
    if ret.returncode == 0:
        print('MS2 topfd running done')
    return 0
    

def flash_align_topfd(ms2_file_path):
    flash_fd_intersec = []
    count = 0 

    #flash low setting
    flashd_mass_file = ms2_file_path +'/MS2_flashdeconv_lowerSetting_output/ms2mass.tsv'
    flashd_spec_ms1_file = ms2_file_path +'/MS2_flashdeconv_lowerSetting_output/ms2specmass_ms1.tsv'
    
    spec_ms1_df = pd.read_csv(flashd_mass_file,sep='\t')
    spec_ms1_df['MonoisotopicMass'] = spec_ms1_df['MonoisotopicMass'].astype(float)   #mass
    spec_ms1_df['MinCharge'] = spec_ms1_df['MinCharge'].astype(float)    #charge
    spec_ms1_df['MaxCharge'] = spec_ms1_df['MaxCharge'].astype(float)
    spec_ms1_df['StartRetentionTime'] = spec_ms1_df['StartRetentionTime'].astype(float)  #scan--rt
    spec_ms1_df['StartRetentionTime'] = spec_ms1_df['EndRetentionTime'].astype(float)
    
    #topfd feature
    fd_mass_file = ms2_file_path +'/topfd/MS2/ms2_spectrum_file/ms2_spectrum_frac.mzrt.csv'
    fd_ms1_df = pd.read_csv(fd_mass_file)
    fd_ms1_df['Mass'] =  fd_ms1_df['Mass'].astype(float)   #mass
    fd_ms1_df['Charge'] =  fd_ms1_df['Charge'].astype(float)   #charge
    fd_ms1_df['rtLo'] =  fd_ms1_df['rtLo'].astype(float)   #scan--rt
    fd_ms1_df['rtHi'] =  fd_ms1_df['rtHi'].astype(float)


    for  _, rt_row in spec_ms1_df.iterrows():
        flash_monoMass = float(rt_row['MonoisotopicMass'])
        flash_rt_start = float(rt_row['StartRetentionTime'])
        flash_rt_end = float(rt_row['EndRetentionTime'])
        flash_charge_min = float(rt_row['MinCharge'])
        flash_charge_max = float(rt_row['MaxCharge'])
        if flash_monoMass == 6250.040891:
            print(flash_monoMass)
            print(flash_rt_start )
            print(flash_charge_max )
        fd_candidate_df = fd_ms1_df.loc[(fd_ms1_df['Mass'] > (flash_monoMass-20)) & (fd_ms1_df['Mass'] < (flash_monoMass+20)) 
                      & (fd_ms1_df['rtLo']*60 <flash_rt_end) & (fd_ms1_df['rtHi']*60 >flash_rt_start)
                      & (fd_ms1_df['Charge'] > (flash_charge_min - 3)) & (fd_ms1_df['Charge'] < (flash_charge_max + 3)),:]
        
        if len(fd_candidate_df)!=0:
            count = count +1
            flash_fd_intersec.append((flash_monoMass,flash_rt_start,flash_rt_end,flash_charge_min, flash_charge_max))
        
        
    flash_fd_intersec.sort()
    for i in flash_fd_intersec:
        print(i)
    print('flashdeconv align with topfd number:',len(flash_fd_intersec))
    return 0
   

def fd_(ms2_file_path):
    
    ms2_js_path = ms2_file_path+'/topfd/MS2/ms2_spectrum_html/topfd/ms1_json/'
    with open(ms2_file_path+'/topfd/MS2/ms2_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Scan', 'Retention Time','ID', 'Mono Mass', 'Charge','MZ Intensity Pairs'])
        
        for j in os.listdir(ms2_js_path):
            #print(j)
            with open(ms2_js_path + j,'r') as uf:
                data = uf.read()
                json_str = data.split('=', 1)[1].strip()
                ms1_data = json.loads(json_str)
                
                for envelope in ms1_data['envelopes']:
                    mz_intensity_pairs = [(peak['mz'], peak['intensity']) for peak in envelope['env_peaks']]
                    writer.writerow([ms1_data['scan'], ms1_data['retention_time'],envelope['id'], envelope['mono_mass'], envelope['charge'], mz_intensity_pairs])
        

    

import pyopenms
import matplotlib.pyplot as plt


def fd_see(ms1_file_path,ms2_file_path,ms2_mzmlFile_name):
    ms1_p_info_2 = ms1_file_path +'/MS1_data_new_gaussian1.csv'
    ms1_p_df_2 = pd.read_csv(ms1_p_info_2)
    resolution = 0.5
    timeTolerance_second = 40   
    scan_time = 5
    mzml_file = ms2_file_path+'/'+ms2_mzmlFile_name
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file,exp)
    
    ms2js_df = pd.read_csv(ms2_file_path+'/topfd/MS2/ms2_data.csv')
    ms2js_df['Retention Time'] = ms2js_df['Retention Time'].astype(float)
    ms2js_df['Mono Mass'] = ms2js_df['Mono Mass'].astype(float)
    ms2js_df['Charge'] = ms2js_df['Charge'].astype(float)
    
    if not os.path.exists(ms1_file_path+'/run_pearson06'):
        os.mkdir(ms1_file_path+'/run_pearson06')
    match_file = ms1_file_path +'/run_pearson06/match_pearson06.csv'
    with open(match_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['precursor id','precursor mono mass','precursor charge','fragment id','fragment mono mass','fragment ms2 scan','fragment ms2 rt','fragment deconv id','fragment charge','fragment intensity','fragment mz internsity pairs'])
    
        precursor_id = 0
        for  _, p_row in ms1_p_df_2.iterrows():
          
            precursor_id = precursor_id +1
            p_mass = float(p_row['Mono Mass'])
            p_id = float(p_row['id'])
            if p_id <81:
                p_charge = eval(p_row['Charge'])
                p_charge_max = max(p_charge)
                p_curve = eval(p_row['gaussian curve'])
                p_curve_y = [i[1] for i in p_curve]
              
                '''
                plt.figure(figsize=(10, 6))
                x_data = [i[0]+scan_time for i in p_curve]
                y_data = [i[1] for i in p_curve]
                plt.plot(x_data, y_data, 'bo',label='Data')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
                '''
        
                ini_candidate_df = ms2js_df.loc[(ms2js_df['Retention Time'] > (p_curve[0][0]-timeTolerance_second )) & (ms2js_df['Retention Time'] <(p_curve[-1][0]+timeTolerance_second )) & (ms2js_df['Mono Mass'] < p_mass) & (ms2js_df['Charge'] < p_charge_max),:]
                
                #plt.figure(figsize=(10, 6))
                fragement_count = 0
                for  _, rt_row in ini_candidate_df.iterrows():    
                    
                    f_rt = float(rt_row['Retention Time'])
                    f_scan = float(rt_row['Scan'])
                    f_deconv_ID = int(float(rt_row['ID']))
                    f_mz = eval(rt_row['MZ Intensity Pairs'])
                    f_mono_Mass = float(rt_row['Mono Mass'])
                    f_charge = float(rt_row['Charge'])
                    
                    f_intensity = 0
                    for i in f_mz:
                        f_intensity = i[1] +f_intensity
                    
                    #print('scan:',scan)
                    #print('f_ID:',f_ID)
                    #print('mono mass:',rt_row['Mono Mass'])
                    #print('Scan:',rt_row['Scan'])
                    if (f_mz[1][0] - f_mz[0][0])>0.7:   
                        mz_interval = Interval(f_mz[0][0]-resolution,f_mz[0][0]+resolution)
                    else:
                        max_tuple = max(f_mz,key=lambda x:x[1])
                        if max_tuple[0]-resolution > f_mz[0][0]:
                            min_i = max_tuple[0]-resolution 
                        else:
                            min_i = f_mz[0][0]
                        if max_tuple[0]+resolution > f_mz[-1][0]:
                            max_i = f_mz[-1][0]
                        else:
                            max_i = max_tuple[0]+resolution
                        
                        mz_interval = Interval(min_i,max_i)
                     
                     
                    time_dependent_intensities1 = []
                    
                    count = 0
                    for spectrum in exp:
                        count = count +1
                        if spectrum.getRT() in Interval(p_curve[0][0]+scan_time-3 ,p_curve[-1][0]+scan_time+3):
                           
                            intensity_sum = 0
                            for peak in spectrum:
                                if peak.getMZ() in  mz_interval :
                                    intensity_sum = intensity_sum +peak.getIntensity() 
                            time_dependent_intensities1.append((spectrum.getRT(),intensity_sum))
                    
                    values = time_dependent_intensities1
                    
                    time_val = [x[0] for x in values]
                    intensity_val = [x[1] for x in values]
                     
                    max_intensiity = max(intensity_val)
                    if max_intensiity >0:
                        scaling_factor = max_intensiity/100
                        
                        if(scaling_factor ==0):
                            print('values:',values)
                            print('scan:',f_scan)
                            print('f_ID :',f_deconv_ID )
                            print('mz_interval')
                        
                        intensity_val = [x[1]/scaling_factor for x in values]
                        
                        
                        r,p = pearson_(p_curve =p_curve_y ,f_curve = intensity_val)
                        if r>0.6:
                            fragement_count =  fragement_count + 1   
                            writer.writerow([precursor_id, p_mass,p_charge_max,fragement_count, f_mono_Mass , f_scan, f_rt, f_deconv_ID, f_charge, f_intensity, rt_row['MZ Intensity Pairs']])
                            
                           
                            '''
                            print(f"Scipy computed Pearson r:{r} and p-value:{p}")
                            print('scan:',f_scan)
                            print('f_ID :',f_deconv_ID )
                            print('mono_Mass ',f_mono_Mass )
                            print("-----------")
                           
                            plt.plot(time_val,intensity_val)
                            plt.xlabel("Retention time (s)")
                            plt.ylabel("Intensity")
                            plt.title("XIC ")
                            # plt.legend()
                            plt.show()
                            '''
                
            
      

    
        
import scipy.stats as stats
def pearson_(p_curve,f_curve):
    EIC1= p_curve
    EIC2 = f_curve
    if len(EIC1) == len(EIC2):
        r,p = stats.pearsonr(EIC1,EIC2)
        
    elif len(EIC1)  < len(EIC2):
        space = len(EIC2) -len(EIC1)
        EIC1.extend([0] * space)
        r,p = stats.pearsonr(EIC1,EIC2)
    else:
        space = len(EIC1) -len(EIC2)
        EIC2.extend([0] * space)
        r,p = stats.pearsonr(EIC1,EIC2)
    
    
    
    return r,p



def check_flat_line():
    flat = False
    return flat

def mapping_to_msalign( ms1_file_path):
    match_file = ms1_file_path + '/run_pearson06/match_pearson06.csv'
    ms1_p_df_2 = pd.read_csv(match_file)  
    precursor_count = int(float(ms1_p_df_2.iloc[-1,0]))
    
    with open(ms1_file_path + '/run_pearson06/match_pearson06_ms2.msalign','a') as f:
        count = 0
        for i in range(1,precursor_count+1):
            f.write('BEGIN IONS\n')
            f.write('ID='+str(count)+'\n')
            f.write('FRACTION_ID=0\nFILE_NAME=D:/abc.mzML\n')
            f.write('SCANS='+str(count)+'\n')
            f.write('RETENTION_TIME='+str(count)+'\n')
            f.write('LEVEL=2\nACTIVATION=HCD\n')
            f.write('MS_ONE_ID='+str(count)+'\n')
            f.write('MS_ONE_SCAN='+str(count)+'\n')
            
            group_rows = ms1_p_df_2.loc[(ms1_p_df_2['precursor id']==i),:]
            p_mono_mass = float(group_rows.iloc[0,1])
            p_charge = float(group_rows.iloc[0,2])
            p_mz =  (p_mono_mass+p_charge) / p_charge
            
            f.write('PRECURSOR_MZ='+str(p_mz)+'\n')
            f.write('PRECURSOR_CHARGE='+str(p_charge)+'\n')
            f.write('PRECURSOR_MASS='+str(p_mono_mass)+'\n')
            f.write('PRECURSOR_INTENSITY='+ str(100000) +'\n')
            
            for  _, rt_row in  group_rows.iterrows():
                f.write(str(rt_row['fragment mono mass'])+'\t'+str(rt_row['fragment intensity'])+'\t'+str(rt_row['fragment charge'])+'\n')
                
            f.write('END IONS')
            f.write('\n\n')
            count=count+1
     
'''
if __name__ == '__main__':
    #RPLC MS2
    ms2_file_path = 'D:/april/deconvolution/unidec_bpi005'
    ms1_file_path = 'D:/april/deconvolution/unidec_bpi005'
    ms2_mzmlFile_name = '20230727RPLC2070_MS2_bpi005.mzML'
    
   
    #MS2_run_lowerSetting_flashdeconv(ms2_file_path,ms2_mzmlFile_name)  
    #MS2_run_topfd(ms2_file_path,ms2_mzmlFile_name)
    
    #flash_align_topfd(ms2_file_path)   
    #fd_(ms2_file_path)
    
    
    
    fd_see(ms1_file_path, ms2_file_path,ms2_mzmlFile_name)
    mapping_to_msalign( ms1_file_path)
'''   
    
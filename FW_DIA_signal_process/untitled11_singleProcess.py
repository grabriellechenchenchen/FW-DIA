# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:25:17 2024

@author: 555
"""

import pandas as pd
import numpy as np
import ast
from intervaltree import Interval
import pyopenms
from bisect import bisect_left, bisect_right

def parse_mz_pairs(mz_str):
    try:
        return ast.literal_eval(mz_str)
    except (ValueError, SyntaxError):
        return []

def load_experiment(mzml_file_path):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file_path, exp)
    return sorted(exp, key=lambda s: s.getRT())

def process_row_single(row, exp_sorted, rt_list, resolution, rt_tolerance):
    f_rt = row['Retention Time']
    f_mz = row['MZ Intensity Pairs']
    
    if len(f_mz) < 2:
        mz_interval = Interval(f_mz[0][0] - resolution, f_mz[0][0] + resolution) if f_mz else Interval(0, 0)
    elif (f_mz[1][0] - f_mz[0][0]) > 0.7:
        mz_interval = Interval(f_mz[0][0] - resolution, f_mz[0][0] + resolution)
    else:
        mz_interval = Interval(f_mz[0][0], f_mz[-1][0])
        
    left = bisect_left(rt_list, f_rt - rt_tolerance)
    right = bisect_right(rt_list, f_rt + rt_tolerance)
    relevant_spectra = exp_sorted[left:right]
    
    intensity_sums = [0] * len(rt_list)
    
    for spectrum in relevant_spectra:
        rt = spectrum.getRT()
        index = bisect_left(rt_list, rt)
        if index < len(rt_list) and rt_list[index] == rt:
            intensity_sum = sum(
                peak.getIntensity() for peak in spectrum if mz_interval.begin <= peak.getMZ() <= mz_interval.end
            )
            intensity_sums[index] = intensity_sum
    
    max_intensity = max(intensity_sums) if intensity_sums else 0
    scaling_factor = max_intensity / 100 if max_intensity > 0 else 1
    normalized_intensities = [i / scaling_factor for i in intensity_sums]
    
    return normalized_intensities


def single_thread_processing(ms2_file_path, ms2_mzmlFile_name, resolution, rt_tolerance):
    ms2js_df = pd.read_csv(ms2_file_path+'/topfd/MS2/ms2_data_filter_scanLimit.csv')  #edit
    
    mzml_file = ms2_file_path + '/' + ms2_mzmlFile_name
    exp_sorted = load_experiment(mzml_file)
    rt_list = [spectrum.getRT() for spectrum in exp_sorted]
    
    ms2js_df = ms2js_df.astype({
        "Scan": float,
        "Retention Time": float,
        'ID': float, 
        'Mono Mass': float, 
        'Charge': float
    })
    
    ms2js_df['MZ Intensity Pairs'] = ms2js_df['MZ Intensity Pairs'].apply(parse_mz_pairs)
    ms2js_df['f_intensity'] = ms2js_df['MZ Intensity Pairs'].apply(lambda mz: sum(i[1] for i in mz))
    

    all_time_intensities = []
    i = 0
    for _, row in ms2js_df.iterrows():
        i+=1
        print(i)
        normalized_intensities = process_row_single(row, exp_sorted, rt_list, resolution, rt_tolerance)
        all_time_intensities.append(normalized_intensities)
    

    rt_columns = pd.DataFrame(all_time_intensities)
    ms2js_df_with_eic = pd.concat([ms2js_df, rt_columns], axis=1)
    
    new_csv_file_path = ms2_file_path + '/topfd/MS2/eic_ms2_data_final.csv'   #edit
    ms2js_df_with_eic.to_csv(new_csv_file_path, index=False)
    
    print(f"Processed data saved to {new_csv_file_path}")




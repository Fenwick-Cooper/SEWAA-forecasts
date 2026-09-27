#!/usr/bin/env python
# coding: utf-8

# From the contents of the directory create the .json file detailing the avaliable dates.
# Year object contains month objects
# Month objects contain day objects
# Day objects contain time objects
# time objects hold a list of valid times

import os
import numpy as np
import json


# The directory with the ELR predictions in
elr_dir = "../interface/ensemble_logistic_regression/ELR_predictions/24h_accumulations/"
country_regiontype = {"Kenya":"subcounty","Ethiopia":"subcounty","Rwanda":"county"}

# Define the sort criteria
def sortFunc(e):
    return e[0]*100*100*100*1000 + e[1]*100*100*1000 + e[2]*100*1000 + e[3]*1000;

for country in ["Kenya","Ethiopia","Rwanda"]:
    elr_dir_country = elr_dir+f"{country}/{country_regiontype[country]}/"
    output_dir = elr_dir_country
    elr_years = []
    times_list = []
    # Create the directory if it is not there
    if not os.path.exists(elr_dir_country):
        os.makedirs(elr_dir_country);
    files = os.listdir(elr_dir_country)
    for file in files:        
        file_name = file
        # "counts_" must be the first part of the file
        if (file_name[0:4] == "GAN_"):
    
            # Is the next part an 8 digit integer, underscore ELR underscore?
            if file_name[4:12].isdecimal() and (file_name[12]=='_') and (file_name[13:16]=='ELR') and (file_name[16]=='_'):
    
                # Start time
                year = int(file_name[4:8])
                month = int(file_name[8:10])
                day = int(file_name[10:12])
    
                # Is the final part of the string "h.nc"
                if file_name.split('v')[-1].split('.nc')[0].isdecimal():
                    valid_day = int(file_name.split('v')[-1].split('.nc')[0])
                    # Include this file in the list
                    times_list.append([year,month,day,valid_day])
    

    
    # Sort the list of times
    times_list.sort(reverse=False, key=sortFunc)
    
    # Make dictionaries
    year = str(times_list[0][0])
    month = str(times_list[0][1])
    day = str(times_list[0][2])
    time = str(times_list[0][3])
    
    # While the start date remains the same
    year_new = year
    month_new = month
    day_new = day
    time_new = time
    idx = 0
    available_dates = {}
    while (idx < len(times_list)):
        valid_times = []
        while (year_new==year) and (month_new==month) and (day_new==day) and (time_new==time):
    
            # Add to the list of valid times at this start date
            valid_times.append(times_list[idx][3])
            idx += 1
    
            # If we haven't reached the end of the list
            if (idx < len(times_list)):
                # Get the time of the next file in times_list
                year_new = str(times_list[idx][0])
                month_new = str(times_list[idx][1])
                day_new = str(times_list[idx][2])
            else:
                # There are no more files in times_list to include
                break
    
        # Build the dictionary
        if year not in available_dates.keys():
            available_dates[year] = {}
        if month not in available_dates[year].keys():
            available_dates[year][month] = {}
        if day not in available_dates[year][month].keys():
            available_dates[year][month][day] = {}
        available_dates[year][month][day] = valid_times
    
        # Move to the next date
        year = year_new
        month = month_new
        day = day_new
        time = time_new
    
    # convert into a JSON string
    available_dates_json = json.dumps(available_dates)
    
    # Write the JSON to a file
    text_file = open(f"{output_dir}/available_dates.json", "w")
    text_file.write(available_dates_json)
    text_file.close()

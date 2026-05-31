#!/usr/bin/env python
# coding: utf-8

# From the contents of the directory create the .json file detailing the avaliable dates.
# Year object contains month objects
# Month objects contain day objects
# Day objects contain time objects

import os
import numpy as np
import json

# The directory with the counts data in
counts_dir = "../interface/view_forecasts/data/counts_s2s"

# Find the years to include
counts_years = []
dirs = os.listdir(counts_dir)
for dir_name in dirs:
    if dir_name.isdigit():
        if (int(dir_name) > 2020) and (int(dir_name) < 3000):   # Optimistic!
            counts_years.append(int(dir_name))

# Where to write the .json file with the list of counts files
output_dir = counts_dir

# The list of all files in counts_dir for all years in counts_years
file_list = []
for i in range(len(counts_years)):
    file_list = file_list + os.listdir(f"{counts_dir}/{counts_years[i]}")

# Extract the dates and times from each compatible file name
times_list = []
for i in range(len(file_list)):
    file_name = file_list[i]

    # Is the file YYYY-MM-DD_histogram_admin1_merged_KeEtRwUg_Xwklead.nc
    if (file_name[0:4].isdecimal() and (file_name[4]=='-') and
        file_name[5:7].isdecimal() and (file_name[7]=='-') and
        file_name[8:10].isdecimal() and
        (file_name[10:44]=='_histogram_admin1_merged_KeEtRwUg_') and
        file_name[44].isdecimal() and
        (file_name[45:54]=='wklead.nc')):

        # Start time
        year = int(file_name[0:4])
        month = int(file_name[5:7])
        day = int(file_name[8:10])

        # Valid time
        valid_week = int(file_name[44])

        # Include this file in the list
        times_list.append([year,month,day,valid_week])

# Define the sort criteria
def sortFunc(e):
    return e[0]*10*100*100 + e[1]*10*100 + e[2]*10 + e[3];

# Sort the list of times
times_list.sort(reverse=False, key=sortFunc)

# Make dictionaries
year = str(times_list[0][0])
month = str(times_list[0][1])
day = str(times_list[0][2])

# While the start date remains the same
year_new = year
month_new = month
day_new = day
idx = 0
available_dates = {}
while (idx < len(times_list)):
    valid_times = []
    while (year_new==year) and (month_new==month) and (day_new==day):

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
    available_dates[year][month][day] = valid_times

    # Move to the next date
    year = year_new
    month = month_new
    day = day_new

# convert into a JSON string
available_dates_json = json.dumps(available_dates)

# Write the JSON to a file
text_file = open(f"{output_dir}/available_dates.json", "w")
text_file.write(available_dates_json)
text_file.close()

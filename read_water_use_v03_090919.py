import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# Version 01: 07312019
# Version 03: 09092019.

  

# Read groundwater level data downloaded from the USGS website
ifile = 'water_use_Idaho.txt'
# only keep the well that have at least 100 measurements
min_number_of_measurements = 0

state = ifile[:-12]
# Read file, line-by-line to find the starting row of data
count = 0
count2 = 0
line_end = 0
well_name = []  # Create an empty list
flag = 0
with open(ifile) as f:
    for line in f:
        count += 1  # line number
#        print(f'Reading line {str(count)}')
#        data = line.split('\t')
        line_content = line.split()

        # Row start the list of wells
        if line_content and line_content[0] == 'state_cd':
#            number_of_sta = int(line_content[5])
#            count2 = count
#            flag = 1
#            line_end = number_of_sta + count
#            print(
#                f'List of water use type is given from line {str(count2+1)} to line {str(line_end)}')
            break

# Read data using pandas
nrow_to_skip = count + 1
col_name = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c10', 'c11', 'c12', 'c28', 'c29', 'c30', 'c46', 'c47', 'c48', 
'c61', 'c62', 'c63', 'c77', 'c78', 'c79', 'c93', 'c94', 'c95', 'c110', 'c111', 'c112', 'c127', 'c128', 'c129', 'c144', 
'c145', 'c146', 'c161', 'c162', 'c163', 'c178', 'c179', 'c180', 'c191', 'c195', 'c196', 'c197', 'c207', 'c208', 'c209', 
'c219', 'c220', 'c221', 'c231', 'c232', 'c233', 'c250', 'c260']
col_name = []
for i in range(283):
    col_name.append('c'+ str(i+1))

df = pd.read_csv(ifile, skiprows=nrow_to_skip,
                 delimiter="\t", names=col_name)

col_keep = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c10', 'c11', 'c12', 'c28', 'c29', 'c30', 'c46', 'c47', 'c48', 
'c61', 'c62', 'c63', 'c77', 'c78', 'c79', 'c93', 'c94', 'c95', 'c110', 'c111', 'c112', 'c127', 'c128', 'c129', 'c144', 
'c145', 'c146', 'c161', 'c162', 'c163', 'c178', 'c179', 'c180', 'c191', 'c195', 'c196', 'c197', 'c207', 'c208', 'c209', 
'c219', 'c220', 'c221', 'c231', 'c232', 'c233', 'c250', 'c260']
df_new = df[col_keep]

df_new.to_csv('testing.csv', index=None)

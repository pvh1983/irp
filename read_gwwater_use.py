import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# Version 01: 07312019
# Version 03: 09092019.

  

# Read the national groundwater use data downloaded from the USGS website
path_to_data = r'C:\\Users\\hpham\\Documents\\P22_IRP\\03_codes\\data\\gw_use\\'
list_of_files = os.listdir(path_to_data)
ifile = path_to_data +  'water_use_Idaho.txt'

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
            break

# Read data using pandas
nrow_to_skip = count + 1
col_name=['c' + str(i+1) for i in range(283)] # list c1 to c283
df_org = pd.read_csv(ifile, skiprows=nrow_to_skip,
                 delimiter="\t", names=col_name)
col_total = ['c12','c30','c48','c63','c79','c95','c112','c129','c146','c163',
             'c180','c191','c197','c209','c221','c233','c250','c260']
df = df_org[col_total]
df = df.apply(pd.to_numeric, errors='coerce') # text to NaN

df.insert(0,'county_nm', df_org.c4)
df.insert(1,'year', df_org.c5)
df.insert(2,'Total_gw', df.sum(axis=1, skipna=True))

df.plot.bar(x='county_nm', y='Total_gw')

df.to_csv('gw_use_by_county.csv')
plt.show()

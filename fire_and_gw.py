import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import datetime as dt
from func_irp import *

domain_to_run = 'ID'          # Choose 'ID' or 'USA'
process_fire_and_gw = True   # Map fires, gwtrends, hydrographs, ...
process_gwuse_data = False     # Map groundwater use [Coming soon ...]

# Versions
# 8/31/2019: Move to sharefile for group coding



# Input files =================================================================
# Groundwater level dataset (USA, shallow wells, at least 10 measurements)
gwfile = 'data_USA/gwlevels_shallow_wells_USA_Monthly_ver01_07172019.csv'  # New data by HPham
# Groundwater level trends before a fire year (using 10 years annual data)
res_file = 'data_USA/GWoutput_USA_10yr_B4fire_10yrminthres.csv' # Calculated trends
# MTBS Burned Areas Boundaries Dataset https://www.mtbs.gov/direct-download
ifile_fire = "zip://./data_USA/mtbs_perimeter_data.zip"
# Watershed boundary
if domain_to_run == 'ID':
    fwsb = r"data_Idaho/WBD/WBDHU8.shp"  # Idahol
elif domain_to_run == 'USA':
    fwsb = r"data_USA/WBD/WBDHU8.shp"  # USA

# Process GW use data USA
path_to_gwuse_files = '' # Coming soon ...

# Output folders to save figures
odri = 'fire_vs_time_all_subbasin_' + domain_to_run




if process_fire_and_gw:

    # Get information of a current domain, output files
    spatial_ext,g,ofile_burn_total,ofile_burn_annua,domain = \
    get_domain_info(dm=domain_to_run)

    # Create a dataframe for annual data
    df_date=gen_date_df(sta_dt=dt.date(1984, 1, 1), end_dt=dt.date(2016, 12, 31))

    # Read MTBS dataset ===========================================================
    fire=read_mtbs(ifile=ifile_fire, spatial_extent=spatial_ext,)

    # Load watershed boundary =====================================================
    print(f'Reading file {fwsb}')
    #ifile_wsb = r"data/WBD_National_GDB.gdb" # irp2 conda need
    ilay = 'WBDHU8'
    subbasin = read_wbd(ifile=fwsb, lay=ilay, exp_shp=False)
    # 2:region; 4:subregion; 6:basin; 8:subbasin; 10:watershed; 12:subwatershed.

    # Load groundwater level trend data
    dftrend, g1, points = load_gwtrends_fr_csv(res_file=res_file)

    # Load montly USGS groundwater levels data ====================================
    # For ploting 1984-2016 (org data from 1940-2019)
    rows_to_drop = list(range(0,528,1)) + list(range(924,953,1))
    dfgwlevels = load_gwlevel_data(ifile=gwfile, rows_to_drop=rows_to_drop)

    # Process subbasin to keep ones that have at least one well
    subbasin = get_basin_with_at_least_one_well(subbasin, g1) # return a df

    # Process subbasin to get total fires in a sub-basin, and fires vs. year
    subbasin, df_annual_burn = get_total_burn(fire, subbasin, df_date)

    # DROP un_burned subbasin =====================================================
    subbasin = drop_unburn_basin(subbasin)



    # PLOT ========================================================================
    print(f'\nPLOTTING ....')

    # Create a folder to save figures
    
    make_new_dir(odri)

    csys = 'epsg:4269'
    n_subbasin = subbasin.shape[0]
    # i=14: Lower Boise
    # i = 15: Payette
    for i in range(n_subbasin):
    #for i in range(14,16,1):
        print(f'i={i}, {subbasin.loc[i].NAME}')
        plot_gwfire_each_basin(subbasin, dftrend, dfgwlevels, fire, domain,
                            csys, points,i,g1,df_annual_burn,odri, bg_layer=True)
        


import geopandas as gpd
import fiona
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import pandas as pd
import scipy as sci
from mpl_toolkits.basemap import Basemap
from datetime import datetime, date, time
#from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import imageio
import datetime as dt

# Define study area
def get_domain_info(dm=''):
    if dm == 'ID':
        #fwsb = r"data_Idaho/WBD/WBDHU8.shp"  # Idahol
        xmin, xmax, ymin, ymax = [-119, -109, 40, 48]  # WBD6 that has Boise
    elif dm == 'USA':    
        #fwsb = r"data/WBD/WBDHU8.shp"  # USA
        #xmin, xmax, ymin, ymax = [-119, -64, 22, 49]  # U.S. Continent
        xmin, xmax, ymin, ymax = [-130, -64, 22, 55]  # U.S. Continent
    spatial_extent = Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin),(xmin, ymin)])
    domain = [xmin, xmax, ymin, ymax]
    g = gpd.GeoSeries([spatial_extent])
    ofile_total_burn = 'total_burn_in_each_subbasin_' + dm + '.csv'
    ofile_yearly_burn = 'yearly_burn_in_each_subbasin_' + dm + '.csv'
    return spatial_extent, g, ofile_total_burn, ofile_yearly_burn, domain

def gen_date_df(sta_dt='', end_dt=''):
    #start_date = dt.date(1984, 1, 1)
    #end_date = dt.date(2016, 12, 31)
    df_dt = pd.DataFrame({'Date': [sta_dt, end_dt], 'Val': [999, 999]})
    df_dt['Date'] = pd.to_datetime(df_dt['Date'])
    df_date = df_dt.resample('A', on='Date').mean()   # resample by month
    df_date = df_date.reset_index()
    df_date = df_date.drop('Val', 1)
    df_date.reset_index()
    return df_date

def read_mtbs(ifile='', spatial_extent=''):
    #fire_burn_thres = 0  # in acres, set to 0 for NO filter
    
    print(f'\nReading mtbs dataset {ifile}')
    fire = gpd.read_file(ifile)
    fire = fire[fire['geometry'].intersects(spatial_extent)]

    Date_to_insert = fire['Year'].map(str) + '/' + fire['StartMonth'].map(str)+ '/' + fire['StartDay'].map(str)
    fire.insert(loc=0, column='Date', value=Date_to_insert)
    fire['Date'] = pd.to_datetime(fire['Date'])
    selected_cols = ['Fire_Name', 'Date', 'Acres', 'geometry']
    fire = fire[selected_cols]
    return fire

# [0]
def make_new_dir(odir):
    if not os.path.exists(odir):  # Make a new directory if not exist
        os.makedirs(odir)
        print(f'Created directory {odir}')

# [1] National watershed boundary
def read_wbd(ifile='', lay='', exp_shp=False):
    # fwsb = r"data\WBD_National_GDB.gdb"  # Windows system
    #fwsb = r"data/WBD_National_GDB.gdb"  # Unix
    print(f'Reading watershed boundary file {ifile}')
    # 2:region; 4:subregion; 6:basin; 8:subbasin; 10:watershed; 12:subwatershed.
    wsb = gpd.read_file(ifile, layer=lay)
    #n_subbasin = wsb.shape[0]
    # wsb.plot()
    # selected_cols = ['AREASQKM', 'NAME', 'STATES', 'geometry']
    # wsb = wsb[selected_cols]
    # wsb_nv = wsb.loc[wsb.STATES == "NV", 'geometry']
    # wsb.head()
    # wsb_nv.plot()

    # Select wsh within the spatial extents
    # wsh_in_domain = wsb[wsb['geometry'].intersects(spatial_extent)]
    # wsh_in_domain.plot(cmap='binary', alpha=0.5)
    if exp_shp:
        ofile='output/' + lay + '.shp'
        wsb.to_file(ofile)
    return wsb

def load_gwtrends_fr_csv(res_file=''):
    print(f'Reading file {res_file}')
    df = pd.read_csv(res_file)
    points = df.apply(lambda row: Point(row.dec_long_va, row.dec_lat_va), axis=1)
    g1 = gpd.GeoSeries(points)
    return df, g1, points

def load_gwlevel_data(ifile='', rows_to_drop=''): # Monthly data, processed after USGS
    print(f'Reading groundwater level input file {ifile}')  
    dfgw = pd.read_csv(ifile)    
    dfgw = dfgw.drop(rows_to_drop)
    return dfgw


def get_basin_with_at_least_one_well(subbasin,g1):
    print(f'Processing subbasins to keep ones that have at least one well')
    basin_keep = []
    n_subbasin = subbasin.shape[0]
    for i in range(n_subbasin):
        subbasin_geo = subbasin.loc[i].geometry
        #print(f'Subbasin: {subbasin.iloc[i].NAME}')
        # Check if any wells in polygon subbasin_geo
        id_points_in_subbasin = g1.within(subbasin_geo)
        count = id_points_in_subbasin.value_counts()
        flag = 1
        try:
            tmp=count[True]
        except:
            flag=0 # meaning no well
        
        # Create a column of flag for dataframe
        if flag==0: # Found no well
            basin_keep.append(0)
        elif flag==1:        
            nwell_in_subbasin=count[True]
            #print(f'i={i}, {subbasin.iloc[i].NAME}: {nwell_in_subbasin} (wells)')
            basin_keep.append(1)

    ps=pd.Series(basin_keep)

    # id of rows that need to drop
    id_row_to_del = ps[ps==0]
    row_to_del = list(id_row_to_del.index)
    # Drop rows
    subbasin = subbasin.drop(subbasin.index[row_to_del])

    # SAVE data (filtered)
    #subbasin.to_csv(of11, index=None)
    
    subbasin = subbasin.reset_index()
    return subbasin

def get_total_burn(fire, subbasin, df_date): # Calculate total burned area
    print(f'\nCalculating total burned area for each sub-basin')
    df_annual_burn = df_date.copy()
    burn_in_subbasin=[]
    n_subbasin = subbasin.shape[0]
    for i in range(n_subbasin):
        subbasin_geo = subbasin.loc[i].geometry
        fire_in_subbasin = fire[fire['geometry'].intersects(subbasin_geo)]      
        nburn_in_subbasin = fire_in_subbasin.shape[0]
        
        # Add a new column to record subbasin's name
        name_subbasin = [subbasin.loc[i].NAME]*nburn_in_subbasin
        fire_in_subbasin.insert(loc=0, column='Subbasin_Name', value=name_subbasin)
        
        df2 = fire_in_subbasin.resample('A', on='Date').sum()
        df2.reset_index(inplace=True)
        new_name = 's' + str(i)
        df2.rename({'Acres':new_name}, axis=1, inplace=True)
        df_merge = pd.merge(df_date, df2, how='left', on=['Date'])

        # Combine all dataframes (one dataframe for each subbasin)
        df_annual_burn = pd.concat([df_annual_burn, df_merge[new_name]], axis=1)

        burn_are_acres = fire_in_subbasin['Acres'].sum()
        burn_in_subbasin.append(burn_are_acres)
        #print(f'i={i}, {subbasin.loc[i].NAME}: {burn_are_acres/1e3} (x1000 acres)')

    subbasin.insert(loc=1, column='Total_Burn', value=burn_in_subbasin)

    # Save data
    #subbasin.to_csv(ofile_total_burn, index=None)
    #df_annual_burn.to_csv(ofile_yearly_burn, index=None)
    
    return subbasin, df_annual_burn

def drop_unburn_basin(subbasin):
    print(f'Processing subbasins to keep ONLY ones that were burned')
    id=subbasin[subbasin['Total_Burn'] == 0]
    id_row_to_del=list(id.index)
    subbasin = subbasin.drop(subbasin.index[id_row_to_del])
    subbasin = subbasin.reset_index()
    
    # Save data?
    #subbasin.to_csv(ofile_total_burn2, index=None)
    return subbasin

def plot_gwfire_background(fig, ax, subbasin, fire, df, csys, points, domain,
                            wsh_lay='', fire_lay='', wel_loc=''):
    #subbasin_geo = subbasin.loc[i].geometry
    xmin, xmax, ymin, ymax = domain

    # PLOT Layer 1 all subbasins (polygons) =======================================
    if wsh_lay:
        subbasin.plot(ax=ax[0, 0], column='Total_Burn',linewidth=0.25, edgecolor = 'black',
                    legend=False, scheme='quantiles', alpha=0.5, zorder=0) # Total_Burn is in 
        subbasin.crs = {'init': csys}
        ax[0, 0].set_title("Total burned area 1984-2016 (x1000 acres)", fontsize=12)
    
    # PLOT Layer 2 fires (polygons) ===========================================
    if fire_lay:
        fire.plot(ax=ax[0, 0], column='Acres', linewidth=0.1, edgecolor = 'r', 
                    legend=False, scheme='quantiles', zorder=1) # Total_Burn is in 
        fire.crs = {'init': csys}

    # PLOT Layer 3 - All shallow wells ======================================================
    if wel_loc:
        gwtrends = gpd.GeoDataFrame(df, geometry=points)
        gwtrends.crs = {'init': csys}
        gwtrends.plot(ax=ax[0, 0], markersize=2, color='b', edgecolors='#bfbfbf', 
                    linewidth=0.15, alpha=0.8,zorder=2)
        ax[0, 0].set_xlim([xmin,xmax])
        ax[0, 0].set_ylim([ymin,ymax])


def plot_gwfire_each_basin(subbasin, dftrend, dfgwlevels, fire, domain,
                           csys, points,i,g1,df_annual_burn, odri, bg_layer=''): # df=dfgwlevels
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize  = (14, 12))
    
    subbasin_geo = subbasin.loc[i].geometry
    n_subbasin = subbasin.shape[0]
    
    # Columns names for subbasins
    col_subbasin = ['s'+ str(id) for id in range(0,n_subbasin+2,1)]
    
    # col_keep for trends col_name
    col_keep = ['site_no'] + ['s' + str(yr) for yr in range(1984,2016,1)]
    
    # Background layers? 
    if bg_layer:
        plot_gwfire_background(fig, ax, subbasin, fire, dftrend, csys, points, domain,
                               wsh_lay=True, fire_lay=True, wel_loc=True)    

    # [1] HIGHLIGHT the current subbasin
    curr_subbasin_poly = subbasin.loc[[i],'geometry']
    curr_subbasin_poly.crs = {'init': csys}
    curr_subbasin_poly.plot(ax=ax[0, 0], linewidth=1, color='None', 
                            edgecolor = 'k', zorder=4) #    

    # [2] PLOT gwtrends before/after each fire year =============================
    id_points_in_subbasin = g1.within(subbasin_geo)
    count = id_points_in_subbasin.value_counts()
    nwell_in_subbasin=count[True]
    id_well = id_points_in_subbasin[id_points_in_subbasin == True]
    df_well = dftrend.iloc[list(id_well.index)]
    
    df_well_to_plot = df_well[col_keep]
    df_well_to_plot = df_well_to_plot.set_index('site_no')
    df_well_to_plot = df_well_to_plot.T
    df_well_to_plot.plot(ax=ax[1, 0], markersize=8, linewidth=0.5, style='o-')
    ax[1, 0].set_title("GW Trends BEFORE a fire year", fontsize=12)
    ax[1, 0].set_ylabel('GW Trend (ft/year)')

    # [3] PLOT GW levels hydrographs ==========================================    
    # Get groundwater levels for wells in current subbasin
    wname = [str(df_well.site_no.tolist()[i]) for i in range(nwell_in_subbasin)]
    wname2 = ['Time'] + wname 
    dfgw_to_plot = dfgwlevels[wname2]
    dfgw_to_plot.Time=pd.to_datetime(dfgw_to_plot['Time'])
    dfgw_to_plot2 = dfgw_to_plot.resample('A', on='Time').mean()   # resample by Year
    dfgw_to_plot2 = dfgw_to_plot2.reset_index()
    dfgw_to_plot2[wname] = -dfgw_to_plot2[wname]

    dfgw_to_plot2.plot(ax=ax[0, 1], x='Time', y=wname, markersize=8, style='o:', 
                       linewidth=0.5) # MOnthly

    ax[0, 1].set_title("Depth to GW level (ft)", fontsize=12)
    ax[0, 1].set_ylabel('GW depth (ft)')   

    # [4] PLOT fires vs. time given a subbasin ====================================
    plot_title = subbasin.iloc[i].NAME + ', Area = ' + str(subbasin['Total_Burn'][i]) + 'Km$^2$'
    
    #Convert acres to sqkm2
    df_annual_burn_normlized = df_annual_burn.copy()    
    df_annual_burn_normlized[col_subbasin]= \
    100*df_annual_burn_normlized[col_subbasin]*0.0040468599998211/subbasin['AREASQKM'][i]
    df_annual_burn_normlized.plot(ax=ax[1, 1], linewidth=0.5,
                        x='Date', y=df_annual_burn_normlized.columns[i+1], 
                        legend=False, style='o--', title=plot_title)
    ax[1, 1].set_ylabel('Burned area (%), normalized to subbasin area') 
    
    # Turn on grid
    ax[0, 1].grid(); ax[1, 0].grid(); ax[1, 1].grid()

    ofile = odri + '/' + subbasin.iloc[i].NAME + '.png'
    #fig = plot1.get_figure()
    print(f'Saving figure {ofile}')
    fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

def plot_fire_gwdepth():
    # PLOT [5] ================================================================
    plot_title = subbasin.iloc[i].NAME + ', Area = ' + str(subbasin['Total_Burn'][i]) + 'Km$^2$'

    ytmp = df_annual_burn[df_annual_burn.columns[i+1]]
    ytmp[3]= 'NaN'
    for ii, wn in enumerate(wname):
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize  = (6, 6))
        xtmp = dfgw_to_plot2[wn]
        ax2.scatter(-xtmp, ytmp, marker='o')
        ax2.set_ylabel('Burned area (acres)')
        ax2.set_xlabel('GW depth (ft)')
        ofile = odri + '/' + subbasin.iloc[ii].NAME + '_'+wn + '.png'
        #fig = plot1.get_figure()
        fig2.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')

    #fig.savefig(ofile, dpi=300, transparent=False, bbox_inches='tight')

    print(f'Saving file {ofile}')


# [6] Read wildfire pologon database
def export_fire_map(ifile='', map_fire_acc='', exp_shp=True):
    xmin, xmax, ymin, ymax = [-119, -109, 40, 48]  # WBD6 that has Boise
    spatial_extent = Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin),(xmin, ymin)])
    g = gpd.GeoSeries([spatial_extent])
    #g.plot()
    #crs = {'init': 'epsg:4326'}
    #polygon_geom = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[spatial_extent])

    get_small_domain = 'ID' # Specify a name of leave it empty ''
    fire_burn_thres = 0  # in acres, set to 0 for NO filter
    ifile = "zip://./data/mtbs_perimeter_data.zip"
    print(f'\nReading file {ifile}')

    fire = gpd.read_file(ifile)
    print(fire.columns)
    if get_small_domain == 'ID':
        print(f'Extract mtbs data for {get_small_domain}')
        selected_cols = ['Fire_Name', 'Year',
                         'StartMonth', 'StartDay', 'Fire_Type', 'Acres', 'geometry']
        fire = fire[selected_cols]
        fire = fire[fire['geometry'].intersects(spatial_extent)]
        fire.plot()

    # Choose fire by burn severity
    fire = fire[fire['Acres'] > fire_burn_thres]

    min_time = fire['Year'].min()
    max_time = fire['Year'].max()

    fire_time = []
    total_burn_in_year = []
    nfire = []
    for k in range(int(min_time), max_time+1):
        if map_fire_acc == 'yearly':
            print(f'Exporting mtbs maps, year-by-year')
            fire_in_year = fire[fire['Year'] == k]
            # print(fire_in_year['Acres'].sum())
            fire_time.append(k)
            total_burn_in_year.append(fire_in_year['Acres'].sum())
            nfire.append(len(fire_in_year))
            
            # create a new directory and file names
            new_dir = 'output_mtbs_yearly_' + get_small_domain
            make_new_dir(new_dir)
            ofile = new_dir + '/fire_year_' + str(k) + '.shp'            
        elif map_fire_acc == 'accum':
            print(f'Exporting fire map, accumulation from year {min_time}')
            fire_in_year = fire[fire['Year'] >= min_time]
            fire_in_year = fire_in_year[fire_in_year['Year'] < k+1]
            
            # create a new directory and file names
            new_dir = 'output_mtbs_accuml_' + get_small_domain
            make_new_dir(new_dir)
            ofile = new_dir + '/fire_year_' + str(k) + '.shp'            
        print(
            f"year: {k}, Nburns: {len(fire_in_year)}, area: {fire_in_year['Acres'].sum()}")
        if exp_shp == True:
            try:
                fire_in_year.to_file(ofile)  # save shapefile
            except:
                print(f'No fire in year: {k} or something went wrong')
    data = {'Date': fire_time, 'Total_burn': total_burn_in_year, 'Nfires': nfire}
    df = pd.DataFrame(data, columns=['Date', 'Total_burn', 'Nfires'])
    ofile = 'burn_stat_' + get_small_domain + '_' + map_fire_acc + '.png'
    df.to_csv(ofile)


def MapPlotNV(df=None, cmap='PiYG', var='', fire_shp_2plt=None, domain_name='',
              fire_year='', start_trend_year=1940, map_fire_acc=False, fill_fire_poly=False,
              vmin=None, vmax=None, gwtrend=True, domain_coor='', wbd_layer_name='', ofile='', fig_dpi=150):

    # fig, ax = plt.subplots(1, 2)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)
    xmin, xmax, ymin, ymax = domain_coor
    if domain_name == 'NV':
        map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax, projection='lcc',
                      resolution='l', lon_0=-98.6, lat_0=39.8, epsg=4269)
    else:
        # print(f'Basemap for {domain_name}')
        map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
                      projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

#    map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
#                  resolution='l', lon_0=-117, lat_0=38, epsg=4269 )
# Center of US = 39.8, 98.6
# Projection options = 'lcc', 'cyl', 'aeqd', 'tmerc', nsper, ortho
#    map = Basemap(llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax, projection='tmerc',
#                  resolution='l', lon_0=-98.6, lat_0=39.8, epsg=4269)
    # map.drawmapscale(0.6, 0.08, 0.25, 0.01, 500, barstyle='fancy')

    # map.bluemarble()  # Background
    # map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)  # Background
    background = False
    if background == True:
        map.arcgisimage(service='World_Shaded_Relief',
                        xpixels=1500, verbose=True)  # Background
#    map.drawmapscale(0.6, 0.08, 0.25, 0.01, 500, barstyle='fancy')

    # pick a  html hex string from this website
    # https://www.w3schools.com/colors/colors_picker.asp
    # grey 75% #bfbfbf; 50%  #808080;

    # County NV
    if domain_name == 'USA':
        state_line_width = 0.4
        state_edge_color = '#808080'
        gw_s_size = 6
        gw_line_width = 0.02
        fire_edge_color = 'r'
        fire_line_width = 0.5
        wbd_line_width = 0.2
        alpha_gw_points = 1        
        path_wbd = 'data/WBD/'  # All States

    else:   # other smaller domain (e.g., a state)
        cousub = False
        if cousub:
            map.readshapefile('data/tl_2015_32_cousub/tl_2015_32_cousub',
                              name='NAME', linewidth=0.5, color='#bfbfbf', drawbounds=True)
        state_line_width = 1.0
        state_edge_color = 'k'
        gw_s_size = 30
        alpha_gw_points = 1
        gw_line_width = 0.1  # for symbols gwlevel trend
        fire_edge_color = 'r'
        fire_line_width = 0.5  # for fire's shapes
        wbd_line_width = 1   # for watershed
        path_wbd = 'data_Idaho/WBD/' # A smaller area

    # STATE

    map.readshapefile('data/esri-us-states/us_states_wgs84',
                      name='state_name', color=state_edge_color, linewidth=state_line_width, drawbounds=True)

    # Map FIRE areas boundaries by year =======================================
    map.readshapefile(fire_shp_2plt, color=fire_edge_color,
                      name='Fire_Name', linewidth=fire_line_width, drawbounds=True)

    if fill_fire_poly == True:
        patches = []
        # for info, shape in zip(map.comarques_info, map.comarques):
        # 30% red is #990000 (darker)
        for info, shape in zip(map.Fire_Name_info, map.Fire_Name):
            patches.append(Polygon(np.array(shape), True))
            ax.add_collection(PatchCollection(patches, facecolor='r',
                                              edgecolor='#990000', linewidths=0.1, zorder=2))

    # Watershed
    if len(wbd_layer_name) == 2:
        wbd_line_wi = [wbd_line_width/5, wbd_line_width/2]
        # WBD_color = ['#0040ff', '#00ff00']  # 50% Blue and Green
        WBD_color = ['green', 'blue']  # 50% Blue and Green
    else:
        WBD_color = '#80ff80'
        wbd_line_wi = wbd_line_width

    if wbd_layer_name != '':
        for i in range(len(wbd_layer_name)):
            ifile_wbd = path_wbd + wbd_layer_name[i]
            print(f'Reading WBD shape file {ifile_wbd}')
            map.readshapefile(ifile_wbd, name='NAME',
                              color=WBD_color[i], linewidth=wbd_line_wi[i], drawbounds=True)
    else:
        print(f'No WBD layers added')
        pass

    # PLOT GROUNDWATER LEVEL LOCATION/TREND ============================================
    x, y = map(df["dec_long_va"].values, df["dec_lat_va"].values)
    if gwtrend == True:
        z_value = df[var]
        s = ax.scatter(x, y, c=z_value, vmin=vmin, vmax=vmax,
                       edgecolors='k', s=gw_s_size, linewidth=gw_line_width, cmap=cmap, alpha=alpha_gw_points)
        # cbar_ax = fig.add_axes([0.15, 0.15, 0.25, 0.01]) # USA
        cbar_ax = fig.add_axes([0.3, 0.08, 0.25, 0.01])  # NV
        cbar = fig.colorbar(s, orientation='horizontal',
                            extend='both', extendfrac='auto', cax=cbar_ax)
        cbar.set_label('Groundwater level trend (ft/yr)',
                       labelpad=0, y=0.08, rotation=0)  # y=0.15 for USA; y = 0.05 for NV
    else:  # Show location only
        s = ax.scatter(x, y, edgecolors='#bfbfbf',
                       s=gw_s_size, linewidth=gw_line_width, alpha=1)

    # PLOT map of groundwater level wells all US ==============================
    # 30% blue #000099, 10% #000033
    map_groundwater_wells = False
    if map_groundwater_wells == True:
        df = gpd.read_file("zip://./data/gwlevel_wells_all_USA.zip")
        x, y = map(df['dec_long_v'], df['dec_lat_va'])
        gw_wells = ax.scatter(x, y, color='g', edgecolors='#000033', s=gw_s_size,
                              linewidth=gw_line_width, alpha=0.75)

    # Export the figure to png ================================================
    if map_fire_acc == 'yearly':
        # ofile = 'fire_year_' + str(fire_year) + '_' + domain_name + '.png'
        ax.set_title(f'BURNED AREAS BOUNDARIES - YEAR: {fire_year}')
    else:
        # ofile = 'acc_fire_year_' + str(fire_year) + '_' + domain_name + '.png'
        ax.set_title(
            f'BURNED AREAS BOUNDARIES FROM {1984} to {fire_year}')
    print(f'Saving map to ... {ofile[-60:]} \n')
    fig.savefig(ofile, dpi=fig_dpi, transparent=False, bbox_inches='tight')
    # fig.savefig(ofile, dpi=200, transparent=False)
    return fig


def gen_ani(indir, ofile):

    frames = []
    # Load each file into a list
    for root, dirs, filenames in os.walk(indir):
        for filename in filenames:
            frames.append(imageio.imread(indir + "/" + filename))

    # Save them as frames into a gif
    # kargs = {'duration': 2}
    kargs = {'fps': 3}
    imageio.mimsave(ofile, frames, 'GIF', **kargs)

    # shutil.move('fire_in_each_year_USA_png.gif', 'C:\Users\hpham\Dropbox\HPham-PLe\IRP2019\03_Codes\output')

    if __name__ == '__main__':
        xmin, xmax, ymin, ymax = [-120.5, -113.5, 34.5, 42.5]
        spatial_extent = Polygon([(xmin, ymax),
                                  (xmax, ymax),
                                  (xmax, ymin),
                                  (xmin, ymin)])
        ifile = "zip://./data/mtbs_perimeter_data.zip"
        export_fire_map(ifile, spatial_extent)

#    plt.show()


def EstimateTrend(df, GW, sta, c_out_name, nyears_thes):
    count = 1
    for id, s in enumerate(sta):     # id is count, s is wname
        #print(f'id = {id}, s = {s} \n')
        data = GW[['Time', s]].copy()
        data.loc[data[s] > 10000, s] = np.nan
        data.loc[data[s] < -10000, s] = np.nan
        data.dropna(inplace=True)
        data['year'] = data['Time'].dt.year

        annualm = data.groupby(['year']).mean()
        annualm.reset_index(inplace=True)
        xm = annualm['year']
        xm = xm.values.reshape(-1, 1)
        ym = annualm[s]
        ym = ym.values.reshape(-1, 1)

        if len(ym) >= nyears_thes:
            #print(f'nyrs = {len(ym)} at station {s}')
            count += 1

        # print(annualm)
        if id == 0:
            print(f'Writing data to column {c_out_name[0]}')
        if annualm.shape[0] >= nyears_thes:
            modelm = sci.stats.theilslopes(ym, xm, 0.95)
            df.loc[id, c_out_name[0]] = modelm[0]
            df.loc[id, c_out_name[1]] = modelm[2]
            df.loc[id, c_out_name[2]] = modelm[3]
#            print(f'trend: {modelm} \n')
#        if id % 500 == 0:
#        print(f'id: {id}')
#            print(
#                f'n_year_obs = {annualm.shape[0]}, trend: {modelm[0]} at id {id}, sta {sta[id]} \n')
    df.index = np.linspace(0, len(sta)-1, len(sta))
#    ofile = c_out_name[0] + '.csv'
    # df.to_csv(ofile)
    print(f'{count-1} station(s) have at least {nyears_thes} years of data \n')
    return df


def InitializeArray(list, nwells):
    num = len(list)
    data = np.nan * np.zeros((num, nwells))
    df = pd.DataFrame(data.T, columns=list)
    return df


def ReadInputData(mf, gf, xmin, xmax, ymin, ymax, yr_of_fire, nyr_to_cal_trend, b4_or_af_fire=''):
    #    gf = 'groundwater_levels_1940_2015.csv'
    #    mf = 'meta.tmp.csv'
    #    res_file = 'GWoutput.csv'
    #    xmin = -120.5
    #    xmax = -113.5
    #    ymin = 34.5
    #    ymax = 42.5
    #    start_fire_time = datetime(2000, 1, 1)
    #    print(f'Processing fire year {yr_of_fire.year} \n')
    if b4_or_af_fire == 'trend_b4':
        gwlevel_start_time = datetime(yr_of_fire.year - nyr_to_cal_trend, 1, 1)
        gwlevel_stop_time = datetime(yr_of_fire.year-1, 12, 31)
    else:
        gwlevel_start_time = yr_of_fire
        gwlevel_stop_time = datetime(
            yr_of_fire.year + nyr_to_cal_trend-1, 12, 31)

    meta = pd.read_csv(mf)
    data = pd.read_csv(gf)

    # Choose wells in a given domain
    meta = meta[meta['dec_lat_va'] < ymax]
    meta = meta[meta['dec_lat_va'] >= ymin]
    meta = meta[meta['dec_long_va'] < xmax]
    meta = meta[meta['dec_long_va'] >= xmin]

    col = meta.site_no
    col = np.ndarray.tolist(col.values)
    col_name = col.copy()
    col_name.insert(0, 'Time')  # Insert 'Time' to the first row of the list
    col_name = [str(i) for i in col_name]  # Convert all elements to str
    col_stations = [str(i) for i in col]  # Convert all elements to str
    data = data[col_name]  # Choose wells in domain
    data['Time'] = pd.to_datetime(data['Time'])
    gwlevel = data[data['Time'] >= gwlevel_start_time]
    gwlevel = gwlevel[gwlevel['Time'] < gwlevel_stop_time]

    print(
        f'Fire_year: {yr_of_fire.year}. GWlevel data: {gwlevel_start_time} to {gwlevel_stop_time}')
    # print(f'Size of data is {data.shape} \n')
    # print(data)
    return meta, gwlevel, col_stations

def process_gwuse():
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
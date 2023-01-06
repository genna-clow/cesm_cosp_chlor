import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
import seaborn as sns

## Define Biome Colors
color_list = np.concatenate((mpl.cm.tab20(range(0,6)), mpl.cm.tab20(range(8,19))), axis = 0)
biomes_cmap = ListedColormap(color_list)
biome_norm = BoundaryNorm(range(1,19), 17, extend='neither')

## Set default font sizes
mpl.rcParams['legend.fontsize'] = 15

def plot_chlor(ds, var, title):
    
    '''
    Plot one timestep of chlorophyll output
    '''

    fig, axs = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180)},
                        figsize=(10,5))

    im1 = axs.pcolormesh(ds.TLONG, ds.TLAT, ds[var], 
                       transform = ccrs.PlateCarree(), vmax = 3, cmap='viridis')

    axs.set_title(title, fontsize = 22)
    axs.add_feature(cartopy.feature.LAND, color='grey', zorder =1)
    axs.coastlines(color = 'black', linewidth = 1)

    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(im1, cax = cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label = 'Surface Chlorophyll [mg/m^3]', size = 18)
    plt.show()
    
    
def plot_clouds(ds, var, title):
    fig, axs = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180)},
                        figsize=(10,5))

    im1 = axs.pcolormesh(ds.TLONG, ds.TLAT, ds[var], 
                       transform = ccrs.PlateCarree(), cmap='Blues_r')

    axs.set_title(title, fontsize = 22)
    axs.add_feature(cartopy.feature.LAND, color='grey', zorder =1)
    axs.coastlines(color = 'black', linewidth = 1)

    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(im1, cax = cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label = 'Total Cloud Cover [%]', size = 18)
    plt.show()
    
def plot_weights(ds, var, title):
    fig, axs = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180)},
                        figsize=(10,5))

    im1 = axs.pcolormesh(ds.TLONG, ds.TLAT, ds[var], 
                       transform = ccrs.PlateCarree(), vmax = 1, cmap='magma')

    axs.set_title(title, fontsize = 22)
    axs.add_feature(cartopy.feature.LAND, color='grey', zorder =1)
    axs.coastlines(color = 'black', linewidth = 1)

    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(im1, cax = cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label = 'Weights', size = 18)
    plt.show()

def pop_add_cyclic(ds):
    
    '''
    Authors: Michael Levy & Matthew Long
    
    '''
    
    nj = ds.TLAT.shape[0]
    ni = ds.TLONG.shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon)    
    lon  = np.concatenate((tlon, tlon + 360.), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)

    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
            
    return dso
    
    
def seasonal_cycle(df_clouds, df_baseline, cloud_lim = [50,100], chlor_lim = [0, 4]):
    '''
    Plot the mean seasonal cycle for cloud-obscured and baseline outputs. 
    The upper subplot shows the missing data and the lower subplot shows the mean chlorophyll concentration.
    
    Parameters
    ----------
    df_clouds : pd.DataFrame
        Datafame with columns 'month', 'percent_missing', and 'mean_chl'. This is the output from the calc_missing_data_cesm function. 
    df_baseline : pd.DataFrame
        Datafame with columns 'month', and 'mean_chl'. This is the output from the calc_missing_data_cesm function. 
    
    ''' 
    
    sns.set(rc={"figure.figsize":(10, 7)})
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,8))

    axes = axes.flatten()

    ax1 = sns.lineplot(x = df_clouds['month'], 
                     y = df_clouds['percent_missing']*100, 
                     estimator = 'mean', linewidth=2, ax = axes[0])

    ax1.set_ylim(cloud_lim)

    ax1.set_yticklabels(ax1.get_yticks(), size = 16)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.set_ylabel("Missing Data (%)", fontsize = 18)


    ax2 = sns.lineplot(data = df_baseline, 
                       x = 'month', y = 'mean_chl', label = 'Baseline',
                       estimator = 'mean', linewidth=2, color = 'black', 
                       alpha = 0.5, ax = axes[1])

    ax2 = sns.lineplot(data = df_clouds, 
                       x = 'month', y = 'mean_chl', label = 'Cloud-Masked',
                     estimator = 'mean', linewidth=2, color = 'green', 
                     alpha = 1, zorder = 1, ax = axes[1])


    ax2.set_xticklabels(ax2.get_xticks(), size = 18)
    date_form = mdates.DateFormatter('%b')
    ax2.xaxis.set_major_formatter(date_form)
    ax2.set_xlabel("")

    ax2.set_ylim(chlor_lim)
    ax2.set_yticklabels(ax2.get_yticks(), size = 16)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylabel("Chlorophyll Conc. (mg/m3)", fontsize = 18)
    ax2.legend(fontsize = 18)
    
    fig.align_ylabels()
    

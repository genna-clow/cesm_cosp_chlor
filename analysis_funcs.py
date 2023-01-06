import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask import delayed


def calc_missing_data_cesm(data, weights, weights_nocld, TAREA, mask, calc_missing = True):
    '''
    Calculate % area missing data from cesm POP output using the TAREA cells.
    
    Parameters
    ----------
    data :
    weights : 
    weights_nocld : 
    biome : 
    calc_missing : 
    
    Output
    ------
    df_final : 
    
    '''

    all_data = []
    
    if calc_missing == True: 
        total_area = TAREA.where(mask, drop = True).sum().compute().values
        # Only calculate missing data during the day -- to align with sat data
        cloudy_wgt = weights/weights_nocld 
        TAREA_wgt_clouds_only = TAREA * cloudy_wgt
    
    # TAREA * weights = area of grid cell seen by satellite
    TAREA_wgt = TAREA * weights
        
    for year in np.unique(data.time.dt.year.values).tolist():
        print(year)
        df_list = []
        
        # Subset datasets by region and year
        cloudy_chlor_reg = data.where(mask, drop = True)\
                               .where(data.time.dt.year==year,drop=True)

        TAREA_wgt_reg = TAREA_wgt.where(mask, drop = True)\
                                 .where(TAREA_wgt.time.dt.year==year,drop=True)
        
        if calc_missing == True: 
            TAREA_wgt_clouds_only_reg = TAREA_wgt_clouds_only.where(mask, drop = True)\
                                    .where(TAREA_wgt_clouds_only.time.dt.year==year,drop=True)
        
        # Loop through days in the year
        for date in cloudy_chlor_reg.time.values:
            if calc_missing == True: 
                df_daily = delayed(calc_daily_missing)(date, cloudy_chlor_reg,
                                                       TAREA_wgt_clouds_only_reg,
                                                       TAREA_wgt_reg, total_area)                
                df_list.append(df_daily)

            else: 
                df_daily = delayed(calc_daily_chlor)(date, cloudy_chlor_reg, 
                                                     TAREA_wgt_reg, total_area)               
                df_list.append(df_daily)
                
        results = dask.compute(df_list)
        df = pd.concat(results[0]).set_index('date')
        all_data.append(df)
    
    df_final = pd.concat(all_data)
    df_final['mean_chl'] = df_final['mean_chl'].astype(float)
    
    return df_final

def calc_daily_missing(date, cloudy_chlor_reg, TAREA_wgt_clouds_only_reg, TAREA_wgt_reg, total_area):
    
    # Select one day of data
    cloudy_chlor_day = cloudy_chlor_reg.sel(time = date)
    TAREA_wgt_clouds_only_day = TAREA_wgt_clouds_only_reg.sel(time = date).fillna(0)
    TAREA_wgt_reg_day = TAREA_wgt_reg.sel(time = date).fillna(0)

    ## Calculate weighted mean chl in region 
    mean_chl = (cloudy_chlor_day*TAREA.where(mask, drop=True)).sum(dim=['nlat','nlon'])/(TAREA_wgt_reg_day).sum(dim=['nlat','nlon']).compute().values

    ## Calculate missing area
    included_area = TAREA_wgt_clouds_only_day.where(cloudy_chlor_day>0).sum().compute().values
    missing_area = total_area - included_area

    ## Percent nan 
    per_nan = missing_area/total_area

    df_daily = pd.DataFrame(data = {'date': [date], 'percent_missing': [per_nan], 'mean_chl': [mean_chl]})
    return df_daily

def calc_daily_chlor(date, cloudy_chlor_reg, TAREA_wgt_reg, total_area):
    
    cloudy_chlor_day = cloudy_chlor_reg.sel(time = date)
    TAREA_wgt_reg_day = TAREA_wgt_reg.sel(time = date).fillna(0)
    
    ## Calculate mean chl in region 
    mean_chl = (cloudy_chlor_day*TAREA.where(mask, drop=True)).sum(dim=['nlat','nlon'])/(TAREA_wgt_reg_day).sum(dim=['nlat','nlon']).compute().values

    df_daily = pd.DataFrame(data = {'date': [date], 'mean_chl': [mean_chl]})
    return df_daily
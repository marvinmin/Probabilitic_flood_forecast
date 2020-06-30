import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import zarr
import geopandas as gpd
import cartopy.crs as ccrs
from shapely.geometry import Point
from geopandas import GeoDataFrame
import geopy.distance
from sklearn.model_selection import train_test_split

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import altair as alt

def create_ploygon_grid(data):
    """
    Takes in an Xarray data array sliced and summed over the time dimension
    creates a grid of shapley polygon objects
    
    Parameters
    ----------
    data: xarray.DataArray
        DataArray summed over time
    
    Return
    ------
    geopandas.GeoDataFrame
        A geodataframe with square polygons that form a grid
    """
    
    # convert xarray to dataframe
    df = data.to_dataframe().reset_index()
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    
    # specify coordinate system
    crs = {'init': 'epsg:4326'}
    
    # create geodataframe of points corresponding to raster
    cells = GeoDataFrame(df, crs=crs, geometry=geometry)
    
    # the geometric manipulations from points to recctangular polygons is 
    # specific to the latitude and longitude since they effect the projection
    # the buffer, skew, scale and rotate values were all choosen visually by plotting
    cells['geometry'] = cells.geometry.buffer(0.145).envelope.skew(-4).scale(xfact=1.65).rotate(4.5)
    return cells

def get_cells_and_distances(shed_gauge_code, grid, watersheds, stations):
    """
    Clips the polygon grid to a watershed based on the watershed code.
    If the watershed contains more than 4 cells, distance from each cell
    to the gauge location is calculated. The cells are then divided into 
    four groups based on distance.
    
    Parameters
    ----------
    shed_gauge_code: str
        str of the shed gauge code. eg '05DC001'
    grid: geopandas.GeoDataFrame
        The grid created with the create_polygon_grid function
    predictor_cols: list of strings
        The list of columns to be lagged and then to train the model
    watersheds: geopandas.GeoDataFrame
        The shapefile of watersheds
    stations: geopandas.GeoDataFrame
        The shapefile of hydrometric stations
    
    Return
    ------
    dict or, dict of dicts
        returns a dictionary for indexing the DataSet object for each geodataframe.
        If there are more than 4 cells in a water shed it returns 5 dict objects within 
        a dict object for indexing. The dict objects are based on the dataframe that is 
        desired
    """
    
    # load the specified watershed and station geopandas object
    shed = watersheds.loc[watersheds['GaugeCode'] == shed_gauge_code].buffer(0)
    station = stations.loc[stations['GaugeCode'] == shed_gauge_code]
    # clip for the given station 
    clipped = gpd.clip(grid, shed)
    
    # determine whether more than 4 NARR cells are captured in the clipp and 
    # process accordingly
    if clipped.shape[0] >= 4:
        # get the distance for each cell from the stations
        distance = []
        for i in range(len(clipped['lat'])):
            dist = geopy.distance.geodesic((np.array(station['Latitude'])[0], np.array(station['Longitude'])[0]),
                                           (np.array(clipped['lat'])[i], np.array(clipped['lon'])[i]))
            distance.append(dist.meters)
        
        # retrieve the distance values that divide the geopandas dataframe into 4 groups
        clipped['distance'] = distance

        _25 = clipped.describe()['distance']['25%']
        _50 = clipped.describe()['distance']['50%']
        _75 = clipped.describe()['distance']['75%']
        
        # use the distance values to slice the dataframe accordingly
        quarters = [clipped.query(f'distance <= {_25}'),
                    clipped.query(f'distance > {_25} & distance <= {_50}'),
                    clipped.query(f'distance > {_50} & distance <= {_75}'),
                    clipped.query(f'distance > {_75}')]
        
        # initialize dict return object
        distance_points = {}
        
        # store all x and y coordinates for every NARR cell captured by the clip,
        # in addition to the proportion of the cell captured by the clip
        distance_points['whole_shed'] = {'df' : clipped[['y', 'x', 'lat', 'lon', 'geometry']],
                   'y': np.array(clipped['y']), 
                   'x': np.array(clipped['x']),
                   'prop' : np.array(clipped.geometry.area / sum(clipped.area))}
        
        # store the x and y points associated with each NARR cell for each 
        # cell captured by the distance grouping
        for i in range(len(quarters)):
            distance_points[f'q{i+1}'] = {'df' : quarters[i][['y', 'x', 'lat', 'lon', 'geometry']],
                           'y': np.array(quarters[i]['y']), 
                           'x': np.array(quarters[i]['x']),
                           'prop' : np.array(quarters[i].geometry.area / sum(quarters[i].area))}
        print("index into dict for dataframe generation:\n['whole_shed'], ['q1'], ['q2'], ['q3'], or ['q4']")
        return distance_points
    
    else: 
        shed_points = {'df' : clipped[['y', 'x', 'lat', 'lon', 'geometry']],
                       'y': np.array(clipped['y']), 
                       'x': np.array(clipped['x']),
                       'prop' : np.array(clipped.geometry.area / sum(clipped.area))}
        print('no distances, watershed to small')
        return shed_points

def plot_distance(gauge_code, full_gauge_dict, stations):
    """
    For all watersheds with greater than 4 cells, plots the distance
    of each cell colour coded. 
    
    Parameters
    ----------
    gauge_code: str
        str of the shed gauge code. eg '05DC001'
    full_gauge_dict: dict
        dictionary with the 4 quartiles of cell distance dataframes
    stations: geopandas.GeoDataFrame
        The shapefile of hydrometric stations

    Return
    ------
    matplotlib.pyplot.plot
        The plot of coloured distance cells
    """

    fig, ax = plt.subplots(1, figsize=(20, 20))
    base1 = full_gauge_dict['q1']['df'].plot(ax=ax, edgecolor='black', color='gold', alpha=0.9)
    base2 = full_gauge_dict['q2']['df'].plot(ax=base1, edgecolor='black', color='yellowgreen', alpha=0.9)
    base3 = full_gauge_dict['q3']['df'].plot(ax=base2, edgecolor='black', color='seagreen', alpha=0.9)
    base4 = full_gauge_dict['q4']['df'].plot(ax=base3, edgecolor='black', color='teal', alpha=0.9)
    station = stations.loc[stations['GaugeCode'] == gauge_code]
    return station.plot(ax=base4, marker='o', color='darkred', markersize=200);

def get_NARR_dataframe(ds, shed_points, time_start='1996-03-01T00:00', time_end='2019-11-01T00:00'):
    """
    Takes and indexing dictionary and calculates a weighted average 
    across a watershed or quarter of a watershed by using cell proportions
    included in the clipped watershed. 
    
    Parameters
    ----------
    ds: xarray.DataSet
        The NARR DataSet
    shed_points: dict
        The indexing dictionary for either an entire watershed 
        or a quarter of it
    time_start: str
        Datetime formatted string to index the DataSet object with
    time_end: str
        Datetime formatted string to index the DataSet object with

    Return
    ------
    pandas.DataFrame
        A data frame spanning the time specified with aggregated
        apcp, ssrun, snom and soilm values. 
    """
    
    # index into the DataSet to retrieve all cells specified in the 
    # clipping dictionary as well as their proportions
    total = []
    for i in range(len(shed_points['y'])):
        total.append(ds.sel(time=slice(time_start, time_end),
                            y=slice(shed_points['y'][i], 
                                    shed_points['y'][i]), 
                            x=slice(shed_points['x'][i], 
                                    shed_points['x'][i])).drop(['x', 
                                                                'y', 
                                                                'lat', 
                                                                'lon', 
                                                                'Lambert_Conformal', 
                                                                'crain']) * shed_points['prop'][i])
    
    # sum all the DataSet objects to get total value since the 
    # weighted average is achieved by multiplying values by their proportions
    averaged = sum(total)
    return averaged.to_dataframe().reset_index().drop(['x', 'y'], axis=1)


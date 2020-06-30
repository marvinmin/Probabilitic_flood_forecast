import numpy as np
import pandas as pd
import datetime
import pickle
import matplotlib.pyplot as plt
%matplotlib inline

import sys
sys.path.insert(1, '..')
from preprocessing import *
from model_evaluation import *
from spatial_functions import *

# Connect to BLOB storage using a shared access signature, and wrap in a memory cache
sas_token = '?sv=2019-10-10&ss=bfqt&srt=sco&sp=rwdlacupx&se=2020-07-02T06:29:39Z&st=2020-05-22T22:29:39Z&spr=https,http&sig=pLsx0Afa18wA6Z9%2BL0YAmHkwYP3FS4ppocKRfKjC1Bo%3D'

blob_store = zarr.ABSStore(
    'narr',
    prefix='NARR.zarr',
    account_name='floodforecasting',
    blob_service_kwargs=dict(sas_token=sas_token)
)
store = zarr.LRUStoreCache(blob_store, max_size=2**28)

# open the xarray DataSet and index into a DataArray to look at the data
ds = xr.open_zarr(store)
da = ds['ssrun']
data = da.sel(time=slice('2004-01-01T00:00', '2004-02-01T00:00'), 
              y=slice(4836987, 5064228), 
              x=slice(4934376, 5226543)).sum(dim='time')

grid = create_ploygon_grid(data)
stations = gpd.read_file('../../data/Shapefiles/Hydrometric_Stations_2020_04_28.shp').to_crs("epsg:4326")
watersheds = gpd.read_file('../../data/Shapefiles/WGS_watersheds.shp')

# Test create_ploygon_grid
assert(grid.geometry.type.unique()[0] == 'Polygon'), "The grid cells should be polygons"

# Test get_cells_and_distances
dict_05DE007 = get_cells_and_distances('05DE007', grid, watersheds, stations)
dict_05DC001 = get_cells_and_distances('05DC001', grid, watersheds, stations)
assert(len(dict_05DE007) == 4), "A small watershed shouldn't have distances in dict"
assert(len(dict_05DC001) == 5), "A large watershed should have distances in dict"

# Test plot_distance
plot1 = plot_distance('05DC001', dict_05DC001, stations)
assert(plot1.numRows == 1), "There should be one plot returned"

# Test get_NARR_dataframe
df = get_NARR_dataframe(ds, dict_05DE007, time_start='1996-03-01T00:00', time_end='1996-03-01T06:00')
assert all(df.columns == ['time', 'apcp', 'snom', 'soilm', 'ssrun']), "The columns of the dataframe is wrong!"
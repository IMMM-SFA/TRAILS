# Source of GIS data: https://chathamncgis.maps.arcgis.com/home/index.html 
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import json
from shapely.geometry.point import Point 
from matplotlib_scalebar.scalebar import ScaleBar

#%% Load data for county information
rt_counties_all = gpd.read_file('RT_Region/Chatham_County_-_County_Boundaries.shp')
counties_in_rt = ['Wake', 'Durham', 'Orange', 'Chatham']
rt_counties_gdf = rt_counties_all[rt_counties_all['NAME'].isin(counties_in_rt)]
rt_crs = rt_counties_gdf.crs

#%% Load data for lake information
chatham_lakes_all = gpd.read_file('RT_Lakes/Chatham_County_-_Hydro_Polygons.shp').to_crs(rt_crs)
chatham_lakes_gdf = chatham_lakes_all[chatham_lakes_all['WATERBODY'] == 'B. EVERETT JORDAN LAKE']
chatham_haw_gdf = chatham_lakes_all[chatham_lakes_all['WATERBODY'] == 'Haw River']

# %%
raleigh_lakes = gpd.read_file('Wake_Lakes/Major_Water_Bodies.shp').to_crs(rt_crs)
raleigh_lakes_names = ['FALLS LAKE', 'LAKE BENSON', 'NEUSE RIVER', 'LAKE CRABTREE', 'LAKE WHEELER']
raleigh_lakes_gdf = raleigh_lakes[raleigh_lakes['NAME'].isin(raleigh_lakes_names)]

#%%
nc_dams = gpd.read_file('North_Carolina_Dam_Inventory.geojson').to_crs(rt_crs)

#%%
dam_list = ['Lake Michie Dam', 'Cane Creek Resevoir Dam','Little River Dam', 'University Lake Dam']
nc_dams_gdf = nc_dams[nc_dams['Dam_Name'].isin(['Lake Michie Dam', 'Cane Creek Resevoir Dam', 
                                                'Little River Dam', 'University Lake Dam'])]
#%%
owasa_service_area_all_gdf = gpd.read_file('OWASAPrimaryServiceArea/WSMPBA_HerHill_Rgwood.shp').to_crs(rt_crs)
owasa_service_area_gdf = owasa_service_area_all_gdf[owasa_service_area_all_gdf['JUR'] == 'OWASA Primary Service Area']
durham_service_area_gdf = gpd.read_file('DurhamPrimaryServiceArea/City_of_Durham_Boundary.shp').to_crs(rt_crs)

#%%
wakecounty_gdf = gpd.read_file('WakeCounty/Corporate_Limits.shp').to_crs(rt_crs)
raleigh_service_area_gdf = wakecounty_gdf[wakecounty_gdf['LONG_NAME'] == 'RALEIGH']
cary_service_area_gdf = wakecounty_gdf[wakecounty_gdf['LONG_NAME'] == 'CARY']
#polygon_raleigh = raleigh_service_area_gdf.geometry.unary_union
#raleigh_combined = gpd.GeoDataFrame(geometry=[polygon_raleigh], crs=rt_crs)
#%%
chatham_service_area = gpd.read_file('ChathamPrimaryServiceArea/Chatham_County_-_Water_Service_Areas.shp').to_crs(rt_crs)
chatham_service_area_gdf = chatham_service_area[chatham_service_area['DISTRICT'] == 'North']

#%%
pittsboro_service_area_all_gdf = gpd.read_file('PittsboroPrimaryServiceArea/Chatham_County_-_Municipal_Boundaries.shp').to_crs(rt_crs)
pittsboro_service_area_gdf = pittsboro_service_area_all_gdf[pittsboro_service_area_all_gdf['NAME'] == 'Pittsboro']

# %%
ax_region = rt_counties_gdf.plot(color='white', edgecolor='grey', linestyle='--', figsize=(15, 15))
owasa_service_area_gdf.plot(ax=ax_region, color='#EF767A', edgecolor='grey', linewidth=1)
durham_service_area_gdf.plot(ax=ax_region, color='#A4CAF3', edgecolor='grey', linewidth=1)
cary_service_area_gdf.plot(ax=ax_region, color='#49BEAA', edgecolor='grey', linewidth=1)
raleigh_service_area_gdf.plot(ax=ax_region, color='#EEB868', linestyle='--', edgecolor='grey', linewidth=1)
chatham_service_area_gdf.plot(ax=ax_region, color='#BFE5B4', alpha=0.7, edgecolor='grey', linewidth=1)
pittsboro_service_area_gdf.plot(ax=ax_region, color='#D4C2FC', edgecolor='grey', linewidth=1)

chatham_lakes_gdf.plot(ax=ax_region, color='#48ACF0', edgecolor='grey')
chatham_haw_gdf.plot(ax=ax_region, color='#48ACF0', edgecolor='#48ACF0', linewidth=2)
raleigh_lakes_gdf.plot(ax=ax_region, color='#48ACF0', edgecolor='grey')

nc_dams_gdf.plot(ax=ax_region, color='navy', edgecolor='navy', markersize=80, marker='o', label='Dams')

ax_region.get_xaxis().set_visible(False)
ax_region.get_yaxis().set_visible(False)

ax_region.spines['top'].set_visible(False)
ax_region.spines['right'].set_visible(False)
ax_region.spines['bottom'].set_visible(False)
ax_region.spines['left'].set_visible(False)

#ax_region.set_title('Research Triangle Region', fontsize=20)
# Create legend handles and labels
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color='#EF767A', label='OWASA'),
    plt.Rectangle((0, 0), 1, 1, color='#A4CAF3', label='Durham'),
    plt.Rectangle((0, 0), 1, 1, color='#49BEAA', label='Cary'),
    plt.Rectangle((0, 0), 1, 1, color='#EEB868', label='Raleigh'),
    plt.Rectangle((0, 0), 1, 1, color='#BFE5B4', alpha=0.7, label='Chatham'),
    plt.Rectangle((0, 0), 1, 1, color='#D4C2FC', label='Pittsboro'),
]

ax_region.add_artist(ScaleBar(0.3048, length_fraction=0.2, dimension="si-length", 
                              units="m"))
# Add legend to the plot
ax_region.legend(handles=legend_handles, loc='lower right', frameon=False, title='Service Areas',
                 prop={'family': 'Gill Sans MT'})
plt.savefig('RT_Region_scalebar.pdf', dpi=300, bbox_inches='tight')


# %% plot the NC state
nc_state_gdf = gpd.read_file('NCDOT_County_Boundaries/NCDOT_County_Boundaries.shp').to_crs(rt_crs)
ax_state = nc_state_gdf.plot(color='white', edgecolor='black', figsize=(15, 15))
rt_counties_gdf.plot(ax=ax_state, color='indianred', edgecolor='grey')
ax_state.get_xaxis().set_visible(False)
ax_state.get_yaxis().set_visible(False)

plt.savefig('NC_State.pdf', dpi=300, bbox_inches='tight')

# %%

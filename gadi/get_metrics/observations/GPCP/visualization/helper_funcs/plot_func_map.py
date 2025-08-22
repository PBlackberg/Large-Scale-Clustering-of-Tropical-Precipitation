'''
# -----------------
#    plot_func
# -----------------

'''

# -- Packages --
import os
import matplotlib
matplotlib.use('Agg')                                                       # makes matplotlib more dask parallelize friendly
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pickle
import numpy as np
import xarray as xr


# == general plotfuncs ==
def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def move_col(ax, moveby):
    ax_position = ax.get_position()
    _, bottom, width, height = ax_position.bounds
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def cbar_ax_below(ds, fig, ax, h):
    ax_position = ax.get_position()
    w = 0.5
    cbar_ax = fig.add_axes([ax_position.x0 + (ax_position.width - ax_position.width * w) / 2,   # left 
                            ax_position.y0 - ds.attrs['cbar_height'] - ds.attrs['cbar_pad'],    # bottom
                            ax_position.width * w,                                              # width
                            ds.attrs['cbar_height']                                             # height
                            ])
    cbar = fig.colorbar(h, cax = cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize = ds.attrs['cbar_numsize'])
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, 
            ax_position.y0 - ds.attrs['cbar_height'] - ds.attrs['cbar_pad'] - ds.attrs['cbar_label_pad'], 
            ds.attrs['cbar_label'], 
            ha = 'center', 
            fontsize = ds.attrs['cbar_fontsize'], 
            transform=fig.transFigure)
    return cbar

def plot_ticks(ds, ax):
    ax.set_xticks(ds.attrs['xticks'], crs=ccrs.PlateCarree()) 
    ax.set_yticks(ds.attrs['yticks'], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_tick_params(labelsize = ds.attrs['xticks_fontsize'])
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_tick_params(labelsize = ds.attrs['yticks_fontsize']) 
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(length = 2)
    ax.xaxis.set_tick_params(width = 1)
    ax.yaxis.set_tick_params(length = 2)
    ax.yaxis.set_tick_params(width = 1)

def plot_xlabel(ds, fig, ax):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2, 
            ax_position.y0 - ds.attrs['xlabel_pad'], 
            ds.attrs['xlabel_label'], 
            ha = 'center', 
            fontsize = ds.attrs['xlabel_fontsize'], 
            transform=fig.transFigure)
    
def plot_ylabel(ds, fig, ax):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 - ds.attrs['ylabel_pad'], 
            ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2, 
            ds.attrs['ylabel_label'], 
            va = 'center', 
            rotation='vertical', 
            fontsize = ds.attrs['ylabel_fontsize'], 
            transform=fig.transFigure)
    
def plot_ax_title(ds, fig, ax):
    ax_position = ax.get_position()
    ax.text(ax_position.x0 + ds.attrs['axtitle_xpad'], 
            ax_position.y1 + ds.attrs['axtitle_ypad'], 
            ds.attrs['axtitle_label'], 
            fontsize = ds.attrs['axtitle_fontsize'], 
            transform=fig.transFigure)

def save_fig(fig, folder, filename):
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path, dpi = 500)
    print(f'plot saved at: {path}')
    plt.close(fig)

def plot_ref_line(axes, ds_line):
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()  # Flatten if it's a 2D array of axes
    else:
        axes = [axes]  
    for dataset, ax in zip(list(ds_line.data_vars.keys()), axes):
        if 'lon' in ds_line.dims:
            lon_values = ds_line['lon']
            lat_values = ds_line[dataset]
            ax.plot(lon_values, lat_values, transform=ccrs.PlateCarree(), color=ds_line.attrs['color'], linestyle="--", dashes=ds_line.attrs['dashes'], linewidth = ds_line.attrs['linewidth'], alpha=1)
        if 'lat' in ds_line.dims:
            lon_values = ds_line[dataset]
            lat_values = ds_line['lat']
            ax.plot(lon_values, lat_values, transform=ccrs.PlateCarree(), color=ds_line.attrs['color'], linestyle="--", dashes=ds_line.attrs['dashes'], linewidth = ds_line.attrs['linewidth'], alpha=1)

def plot_settings(ds, title = '', threshold = 0, vmin = None, vmax = None, cmap = 'Blues'):
    ''' format plot '''
    width, height = 8.5, 4                                                                                  # size of figure
    width, height = [f / 2.54 for f in [width, height]]                                                     # convert to inches
    ds.attrs.update({
        # -- Figure -- 
        'width': width, 'height': height, 'nrows': 1, 'ncols': 1,
        'axtitle_label': f'{title}','axtitle_xpad': 0.01, 'axtitle_ypad': 0.045, 'axtitle_fontsize': 8,
        # -- plot -- 
        'name': 'var', 'cmap': cmap, 'vmin': vmin,'vmax': vmax,'threshold': threshold,
        # -- axes -- 
        'scale': 1.1, 'move_row': 0.15, 'move_col': 0,
        'yticks_fontsize':  8, 'ylabel_pad': 0.065, 'ylabel_label': '', 'ylabel_fontsize':  8, 'yticks': [-20, 0, 20],
        'xticks_fontsize':  8, 'xlabel_pad': 0.15, 'xlabel_label': '', 'xlabel_fontsize':  8, 'xticks': [60, 120, 180, 240, 300],
        # -- colorbar --
        'cbar_height': 0.035, 'cbar_pad': 0.2,'cbar_label_pad': 0.2,'cbar_label': r'LCF [%]', 'cbar_numsize': 8, 'cbar_fontsize': 8,
        })
    return ds

def ds_contour_settings(ds):
    ''' format plot '''
    ds.attrs.update({
        # -- contour --
        'name': 'var', 'threshold': [ds["var"].quantile(0.5)], 'color': 'k', 'linewidth': 0.5,
        })
    return ds

# == main ==
def plot(da, temp_path, lines = [], ds_ontop = None, ds_ontop2 = None, ds_contour = None, title = ''):
    # -- put data in dataset, and give specs --
    ds = xr.Dataset({'var': da})
    ds = plot_settings(ds, vmin = 0, vmax = 17.5, title = title)
    ds_contour = ds_contour_settings(ds_contour)

    # -- create figure --
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, ax = plt.subplots(ds.attrs['nrows'], ds.attrs['ncols'], figsize = (ds.attrs['width'], ds.attrs['height']), subplot_kw = dict(projection=projection))
    lat, lon = ds.lat, ds.lon
    lonm,latm = np.meshgrid(lon, lat)
    ax.coastlines(resolution="110m", linewidth=0.6)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

    # -- plot contour --
    contours = ax.contour(lonm, latm, ds_contour[ds_contour.attrs['name']], 
                levels =        ds_contour.attrs['threshold'],
                colors =        'k', transform=ccrs.PlateCarree(),
                linewidths =    0.5)

    # -- plot data --
    name = list(ds.data_vars)[0]
    h_pcm = ax.pcolormesh(lonm, latm, ds[name], 
                            transform=ccrs.PlateCarree(),
                            cmap = ds.attrs['cmap'], 
                            vmin = ds.attrs['vmin'], 
                            vmax = ds.attrs['vmax'])
    name_ontop = list(ds_ontop.data_vars)[0]
    da_ontop = ds_ontop[name_ontop]
    ax.pcolormesh(lonm, latm, da_ontop, 
                transform=ccrs.PlateCarree(),
                cmap = 'Greys', 
                vmin = 0, 
                vmax = 1)
    name_ontop = list(ds_ontop2.data_vars)[0]
    da_ontop = ds_ontop2[name_ontop].where(np.isnan(da), np.nan)
    h_pcm = ax.pcolormesh(lonm, latm, da_ontop, 
                transform=ccrs.PlateCarree(),
                cmap = 'Reds', 
                vmin = 0, 
                vmax = 100)

    # -- plot reference line --
    for ds_line in lines:
        plot_ref_line(ax, ds_line)

    # -- format axes --
    scale_ax(ax, ds.attrs['scale'])
    move_row(ax, ds.attrs['move_row'])     
    move_col(ax, ds.attrs['move_col'])
    cbar_ax_below(ds, fig, ax, h_pcm)
    plot_ticks(ds, ax)
    plot_ylabel(ds, fig, ax)
    plot_xlabel(ds, fig, ax)
    plot_ax_title(ds, fig, ax)
    folder = os.path.dirname(temp_path)
    filename = os.path.basename(temp_path)
    save_fig(fig, folder, filename)
    return fig




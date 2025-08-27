'''
# -----------------
#  plot_func_map
# -----------------

'''

# -- Packages --
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np


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
    cbar_ax = fig.add_axes([ax_position.x0,                                                     # left 
                            ax_position.y0 - ds.attrs['cbar_height'] - ds.attrs['cbar_pad'],    # bottom
                            ax_position.width,                                                  # width
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
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)


# == main ==
def plot_snapshot(ds, temp_path):
    # -- create figure --
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, ax = plt.subplots(ds.attrs['nrows'], ds.attrs['ncols'], figsize = (ds.attrs['width'], ds.attrs['height']), subplot_kw = dict(projection=projection))
    lat, lon = ds.lat, ds.lon                                                
    lonm,latm = np.meshgrid(lon, lat)
    ax.coastlines(resolution="110m", linewidth=0.6)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    # -- plot data --
    name = list(ds.data_vars)[0]
    h_pcm = ax.pcolormesh(lonm, latm, ds[name], 
                            transform=ccrs.PlateCarree(),
                            cmap = ds.attrs['cmap'], 
                            vmin = ds.attrs['vmin'], 
                            vmax = ds.attrs['vmax'])
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



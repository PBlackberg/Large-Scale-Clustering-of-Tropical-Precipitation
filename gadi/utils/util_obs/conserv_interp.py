'''
# ----------------------------
#  Conservative interpolation
# ----------------------------

'''

# -- Packages --
import numpy as np
import xarray as xr
import subprocess
from pathlib import Path
from tempfile import (NamedTemporaryFile, TemporaryDirectory)
import os
import sys
from subprocess import run, PIPE

# -- Imported scripts --
sys.path.insert(0, os.getcwd())
import utils.user_specs as mS


def run_cmd(cmd, path_extra=Path(sys.exec_prefix) / "bin"):
    ''' Run a bash command (terminal command) and capture output (this function is also available in myFuncs_dask)'''
    env_extra = os.environ.copy()
    env_extra["PATH"] = str(path_extra) + ":" + env_extra["PATH"]
    status = run(cmd, check=False, stderr=PIPE, stdout=PIPE, env=env_extra)
    if status.returncode != 0:
        error = f"""{' '.join(cmd)}: {status.stderr.decode('utf-8')}"""
        raise RuntimeError(f"{error}")
    return status.stdout.decode("utf-8")

def grid_description(x_res, y_res, switch_area, full_domain):
    if full_domain:
        xsize = int(360 / x_res)
        ysize = int(180 / y_res)
        xfirst = x_res / 2
        yfirst = -90 + y_res / 2
    else:
        xsize = int(360 / x_res)
        ysize = int(180 / y_res)
        xfirst = x_res / 2
        yfirst = -90 + y_res / 2

    content = f"""gridtype = lonlat
gridsize = {xsize * ysize}
xsize = {xsize}
ysize = {ysize}
xfirst = {xfirst}
xinc = {x_res}
yfirst = {yfirst}
yinc = {y_res}
"""
    return content

def get_weights(da_in, simulation_id, x_res, y_res, folder_scratch, path_targetGrid, switch_area, full_domain):
    folder = Path(folder_scratch) / 'temp_process' / 'weights' / f'{simulation_id}_weights_conserv_latlon_{int(360 / x_res)}_{int(180 / y_res)}'
    if folder.exists(): # only recreate weights if needed
        path_gridDes = folder / "gridDescription.txt"        
        path_weights = folder / "weights.nc"
    else:
        folder.mkdir(parents=True, exist_ok=True)
        path_gridDes = folder / "gridDescription.txt"
        path_weights = folder / "weights.nc"
        with path_gridDes.open("w") as f:
            f.write(grid_description(x_res, y_res, switch_area, full_domain))  
        with NamedTemporaryFile() as temp_file:
            path_dsIn = Path(temp_file.name)
            da_in.to_netcdf(path_dsIn, mode="w")
            command = [
                "cdo",
                f"gencon,{path_gridDes}",
                str(path_dsIn),
                str(path_weights),
                ]
            run_cmd(command)
            # subprocess.run(command, check=True)
    return path_gridDes, path_weights

def convert_grid(ds_in, folder_scratch, path_gridDes, path_weights, path_targetGrid):
    temp_dir = TemporaryDirectory(dir=f'{folder_scratch}/temp_process', prefix="remap_conserv_")
    # try:
    path_dsOut, path_dsIn = Path(temp_dir.name) / "ds_out.nc", Path(temp_dir.name) / "ds_in.nc"
    ds_in.to_netcdf(path_dsIn, mode="w")  # Write the data to a temporary netcdf file
    command = [
        "cdo",
        f"remapcon,{path_gridDes}",
        str(path_dsIn),
        str(path_dsOut)
    ]
    run_cmd(command)
    # subprocess.run(command, check=True)
    ds_out = xr.open_dataset(path_dsOut).load()
    return ds_out
    # finally:
    #     temp_dir.cleanup()  # Ensure temporary files are removed

def conservatively_interpolate(da_in, res, switch_area, simulation_id, full_domain = True):
    _, folder_scratch, _, _, _ = mS.get_user_specs()
    path_targetGrid = ''    # could maybe include the conversion form icon grid to latlon grid too
    # path_targetGrid = '/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc'
    # ds = xr.open_dataset(path_targetGrid, engine="netcdf4").rename({"cell": "ncells"})    # 4.82 GB (grid info)
    # da_in = da_in.assign_coords(clon=("ncells", ds.clon.data * 180 / np.pi), clat=("ncells", ds.clat.data * 180 / np.pi))

    # -- weights --
    x_res = res
    y_res = res
    path_gridDes, path_weights = get_weights(da_in, simulation_id, x_res, y_res, folder_scratch, path_targetGrid, switch_area, full_domain)

    # -- grid conversion --
    ds_in = xr.Dataset(data_vars = {da_in.name: da_in}) 
    ds_out = convert_grid(ds_in, folder_scratch, path_gridDes, path_weights, path_targetGrid)
    return ds_out[da_in.name]
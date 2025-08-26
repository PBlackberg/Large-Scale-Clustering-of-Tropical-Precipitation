''' 
# ----------------
#   ERA5 data
# ----------------
ERA5 - ECMWF Re-Analysis v5
https://ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
'''

# == imports ==
# -- packages --
import xarray as xr
import numpy as np
import glob

# -- imported scripts --
import os
import sys
sys.path.insert(0, os.getcwd())
import utils.util_obs.conserv_interp    as cI


# == get raw data ==
# -- convert name --
def convert_to_era_var_name(var_name):
    # 3D variables (pressure-levels)
    var_nameERA = 'r'       if var_name == 'hur'    else None
    var_nameERA = 'q'       if var_name == 'hus'    else var_nameERA
    var_nameERA = 't'       if var_name == 'ta'     else var_nameERA
    var_nameERA = 'z'       if var_name == 'zg'     else var_nameERA
    var_nameERA = 'w'       if var_name == 'wap'    else var_nameERA
    var_nameERA = 'cc'      if var_name == 'cl'     else var_nameERA
    # 2D variables (single-levels)
    var_nameERA = 'slhf'    if var_name == 'hfls'   else var_nameERA
    var_nameERA = 'sshf'    if var_name == 'hfss'   else var_nameERA

    var_nameERA = 'ttr'     if var_name == 'rlut'   else var_nameERA    # net TOA (no downwelling, so same as outgoing)
    var_nameERA = 'str'     if var_name == 'rls'    else var_nameERA    # net surface (into surface) 
    var_nameERA = 'strd'    if var_name == 'rlds'   else var_nameERA    # downwelling surface
    var_nameERA = 'stru'    if var_name == 'rlus'   else var_nameERA    # upwelling is calculated from net surface and downwelling surface (rlus = rlds - rls)

    var_nameERA = 'tsr'     if var_name == 'rst'    else var_nameERA    # net TOA (into atmosphere)
    var_nameERA = 'tisr'    if var_name == 'rsdt'   else var_nameERA    # downwelling TOA
    var_nameERA = 'tsru'    if var_name == 'rsut'   else var_nameERA    # upwelling is calculated from net TOA and downwelling TOA (rsut = rsdt - rst)
    var_nameERA = 'ssr'     if var_name == 'rss'    else var_nameERA    # net surface (into surface)
    var_nameERA = 'ssrd'    if var_name == 'rsds'   else var_nameERA    # downwelling surface
    var_nameERA = 'ssru'    if var_name == 'rsus'   else var_nameERA    # upwelling is calculated from net surface and downwelling surface (rsus = rsds - rss)

    var_nameERA = '2t'      if var_name == 'tas'    else var_nameERA 
    return var_nameERA

# -- get files --
def get_files(var_name, var_nameERA, t_freq, year):
    # -- determine time freq folder --
    if t_freq == 'monthly':
        era_type = 'monthly-averaged'
    elif t_freq == 'daily':
        era_type = 'reanalysis' 
    # -- determine data dimension folder --
    if var_name in ['hur', 'hus', 'ta', 'zg', 'wap', 'cl']:
        lavel_type = 'pressure-levels'
    else:
        lavel_type = 'single-levels'
    var_nameERA = convert_to_era_var_name(var_name)
    folder = f'/g/data/rt52/era5/{lavel_type}/{era_type}/{var_nameERA}/{year}'
    files = [f for f in os.listdir(folder)]
    files = sorted(files, key=lambda x: x.split("_")[-1].split(".")[0])
    return folder, files

# == pre-process ==
# -- convert units --
def convert_units(da, var_nameERA, t_freq):
    # -- coordinates --
    da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = da.sortby('lat')
    if 'level' in da.dims:
        da['level'] = da['level']*100                                           # convert from millibar (hPa) to Pa
        da = da.rename({'level': 'plev'})
        da = da.sortby('plev', ascending = False)
    # -- daily --
    if t_freq == 'daily' and var_nameERA in ['slhf', 'sshf', 'strd', 'str', 'ttr', 'tisr', 'ssrd' , 'ssr', 'tsr']:
        da = - da / (60 * 60 * 24)                                              # convert J/m^2 to W/m^2 (seems to be daily average value) Power (W) = Energy / Time  (rate/s), Energy = J/m^2 (direction is also positive in cmip and negative in ERA5 for rlut)
    # -- monthly --
    elif t_freq == 'monthly' and var_nameERA in ['ttr', 'str', 'strd', 'tsr', 'tisr', 'ssr', 'ssrd', 'slhf', 'sshf', 'rss']:
        da = da / (60 * 60 * 24)                                                # convert J/m^2 to W/m^2 Power (W) = Energy / Time  (rate)(s^-1), Energy = J/m^2
    elif t_freq == 'monthly' and var_nameERA in ['ttr']:                        # negative in ERA5, but positive in CMIP (ERA is net flux into atmosphere, whereas CMIP is upwelling flux)
        da = - da                                                               #
    elif t_freq == 'monthly' and var_nameERA in ['slhf', 'sshf']:               #
        da = - da                                                               #
    elif t_freq == 'monthly' and var_nameERA in ['w']:                          #
        da = da * 60 * 60 * 24 / 100                                            # from pa/s to hpa/day
    elif t_freq == 'monthly' and var_nameERA in ['cc']:                         #
        da = da * 100                                                           # cloud fraction as percentage    
    elif t_freq == 'monthly' and var_nameERA in ['q', 'r', 't', 'z']:           #
        pass
    else:
        print('variable not recognized')
        print(da)
        print('check units for unit conversion')
        print('exiting')
        exit()
    return da

# -- pre-process --
def pre_process(ds, var_nameERA, t_freq, dataset, regrid_resolution):
    # -- pick variable --
    da = ds[var_nameERA].load()
    # -- convert units and coodinates --
    da = convert_units(da, var_nameERA, t_freq)
    # -- get timefreq --
    if t_freq == 'daily':
        da = da.resample(time='1D').mean()
    elif t_freq == 'monthly':
        da = da.resample(time='MS').mean()
    # -- regrid --
    da = cI.conservatively_interpolate(da_in =              da.load(), 
                                        res =               regrid_resolution, 
                                        switch_area =       None,                      # regrids the whole globe for the moment 
                                        simulation_id =     dataset
                                        )
    return da

# == handle time sections ==
def get_timesections(n_jobs, time_period):
    year1, month1 = map(int, time_period.split(':')[0].split('-'))                                                                  #
    year2, month2 = map(int, time_period.split(':')[1].split('-'))                                                                  #
    timesteps = [(year, month) for year in range(int(year1), int(year2) + 1) for month in range(1, 13)                              # year, month pair
                 if not (year == year1 and month < month1) and not (year == year2 and month > month2)]                              # clipping months of first and last year
    time_sections = np.array_split(timesteps, n_jobs)                                                                               #
    return time_sections

# == process one month at a time ==
def get_one_month(var, t_freq, dataset, resolution, year, month, process_data_further):
    # -- get files --
    var_nameERA = convert_to_era_var_name(var)
    folder, files = get_files(var, var_nameERA, t_freq, year)
    pattern = f"*{year}{month:02}*"
    files = glob.glob(f"{folder}/{pattern}")

    # -- concatenate --
    ds = xr.open_mfdataset(files, combine='by_coords', parallel = True)

    # -- pre-process --
    da = pre_process(ds, var_nameERA, t_freq, dataset, regrid_resolution = resolution)

    # -- custom process --
    da = process_data_further(da)
    return da


# == get data ==
def get_data(process_request, process_data_further):
    var, dataset, t_freq, resolution, time_period, year, month = process_request
    if time_period:
        da_list = []
        time_section = get_timesections(n_jobs = 1, time_period = time_period)[0] 
        years, months = zip(*time_section)       
        for idx, (year, month) in enumerate(zip(years, months)):         
            da_month = get_one_month(var, t_freq, dataset, resolution, year, month, process_data_further)
            da_list.append(da_month)
        da = xr.concat(da_list, dim = 'time')
    else:
        da = get_one_month(var, t_freq, dataset, resolution, year, month)
    return da


# == when this script is ran ==
if __name__ == '__main__':    
    dataset = 'ERA5'
    var = 'hus'
    t_freq = 'monthly'
    resolution = 2.8
    time_period = '1998-01:2022-12'     # if this is not given:
    year = '1998'                       # then it picks out this
    month = '1'                         #
    process_request = [var, dataset, t_freq, resolution, time_period, year, month]
    def process_data_further(da):
        ''
        return da
    da = get_data(process_request, process_data_further)    
    print(da)

    # ERA5 variables
    # Single level variables:
    # 100u    asn   csf    fal    istl2   lsm     mgwd    msdrswrf    msqs       mwd3   pp1d   smlt   str    tclw    tsrc   viiwn  vioze   viwvn
    # 100v    awh   csfr   fdir   istl3   lsp     mgws    msdrswrfcs  msr        mwp    ptype  sp     strc   tco3    ttr    vike   viozn   vst
    # 10fg    bfi   cvh    flsr   istl4   lspf    mlspf   msdwlwrf    msror      mwp1   rhoao  src    strd   tcrw    ttrc   viked  vipie   wdw
    # 10u     bld   cvl    fsr    kx      lsrr    mlspr   msdwlwrfcs  msshf      mwp2   ro     sro    strdc  tcslw   tvh    vikee  vipile  wind
    # 10v     blh   dctb   gwd    lai-hv  lssfr   mlssr   msdwswrf    mssror     mwp3   rsn    sshf   swh    tcsw    tvl    viken  vit     wmb
    # 140251  cape  deg0l  hcc    lai-lv  ltlt    mn2t    msdwswrfcs  mtdwswrf   mx2t   sd     ssr    swh1   tcw     u10n   vilwd  vithe   wsk
    # 2d      cbh   dl     hmax   lblt    mbld    mngwss  msdwuvrf    mtnlwrf    mxtpr  sdfor  ssrc   swh2   tcwv    ust    vilwe  vithed  wsp
    # 2t      cdir  dndza  i10fg  lcc     mcc     mntpr   mser        mtnlwrfcs  nsss   sdor   ssrd   swh3   tisr    uvb    vilwn  vithee  wss
    # acwh    cdww  dndzn  ie     lgws    mcpr    mntss   msl         mtnswrf    p1ps   sf     ssrdc  swvl1  tmax    v10n   vima   vithen  wstar
    # alnid   chnk  dwi    iews   licd    mcsr    mp1     mslhf       mtnswrfcs  p1ww   shts   ssro   swvl2  totalx  viec   vimad  vitoe   z
    # alnip   ci    dwps   ilspf  lict    mdts    mp2     msmr        mtpr       p2ps   shww   sst    swvl3  tp      vigd   vimae  vitoed  zust
    # aluvd   cin   dwww   inss   lmld    mdww    mper    msnlwrf     mvimd      p2ww   skt    stl1   swvl4  tplb    vige   viman  vitoee
    # aluvp   cl    e      ishf   lmlt    megwss  mpts    msnlwrfcs   mwd        pev    slhf   stl2   tauoc  tplt    vign   vimat  vitoen
    # anor    cp    es     isor   lsf     mer     mpww    msnswrf     mwd1       phiaw  slor   stl3   tcc    tsn     viiwd  vimd   viwvd
    # arrc    crr   ewss   istl1  lshf    metss   mror    msnswrfcs   mwd2       phioc  slt    stl4   tciw   tsr     viiwe  viozd  viwve

    # Pressure level variables:
    # cc  ciwc  clwc  crwc  cswc  d  o3  pv  q  r  t  u  v  vo  w  z


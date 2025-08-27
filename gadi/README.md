# gadi
<br>
<img src='logo.png' align="right" height="139" />
This repository includes example scripts to fetch and pre-process data and generate metrics from observations and Global Climate Models (GCMs) participating the in Coupled Model Inter-comparison project phase 6 (CMIP6).

### Repository structure
```bash
├── gadi/
    ├── get_metrics/
    │   ├── models/
    │   │   └── cmip/...
    │   │       └── doc_metrics/...
    │   │           └── mean_area/
    │   │               ├── calc_metric.py
    │   │               ├── main_func.py
    │   │               └── submit_as_job.py
    │   └── observations/
    │      └── GPCP/...
    │           ├── visualization/
    │           │   ├── helper_funcs/
    │           │   │   └── plot_func_map.py
    │           │   ├── plots/...
    │           │   │   └── conv_itcz_cl_0.png
    │           │   ├── plots2/...
    │           │   │   └── conv_itcz_cl_0.svg
    │           │   ├── calc_metric.py
    │           │   ├── main_func.py
    │           │   └── submit_as_job.py
    │           └── conv/
    │               └── conv_map_correlation/
    │                   ├── calc_metric.py
    │                   ├── main_func.py
    │                   └── submit_as_job.py
    ├── utils/...
    │    ├── util_calc/...
    │    │   └── doc_metrics/...
    │    │       └── mean_area/
    │    │           └── mean_area.py
    │    ├── util_cmip/...
    │    │   ├── get_cmip_data.py
    │    │   └── ecs_data.py
    │    ├── util_obs/...
    │    │   └── get_GPCP_data.py
    │    ├── util_qsub/
    │    │   ├── interactive_script.py
    │    │   └── submission_funcs.py
    │    └── user_specs.py
    └── environment.yml
```

### How to use repository
First, change the paths in utils/user_specs.py such that the scripts know from where to save/load data and metrics. <br>
Next, change the working directory to where the "gadi" folder is. This is necessary as all scripts import the "utils" folder from the current working directory. <br>
To submit a metric calculation as a HPC job, run "submit_as_job.py". This script includes customizable settings for the calculation such as domain area, resolution, etc. <br>
To run metric calculation interactively, run main_func.py. This script still use the settings specified in "submit_as_job.py". <br>
If timestep data is saved, "calc_metric.py" can isolate the targeted calculation for quick plotting / printing. <br>


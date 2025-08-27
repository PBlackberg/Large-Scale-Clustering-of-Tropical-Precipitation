# levante
<br>
<img src='logo.png' align="right" height="139" />
This repository includes example scripts to fetch and pre-process data from IFS 9 FESOM 5. Metrics are calculated in the same way as demonstrated in the "gadi" folder.

### Repository structure
```bash
└── levante/..
    ├── get_data/..
    │   └── models/
    │       ├── ifs_low_res/
    │           └── vars_2d/
    │               └── pr_data/
    │                   ├── helper_funcs/
    │                   │   └── plot_func_map.py
    │                   ├── plots/
    │                   │   └── daily_field.png
    │                   ├── calc_metric.py
    │                   ├── main_func.py
    │                   └── submit_as_job.p
    ├── utils/..
    │   └── user_specs.py
    └── environment.yml
```

### How to use repository
First, change the paths in utils/user_specs.py such that the scripts know from where to save/load data and metrics. <br>
Next, change the working directory to where the "levante" folder is. This is necessary as all scripts import the "utils" folder from the current working directory. <br>
To submit a data generation / metric calculation as a HPC job, run "submit_as_job.py". This script includes customizable settings for the calculation such as domain area, resolution, etc. <br>
To run data generation / metric calculation interactively, run main_func.py. This script still use the settings specified in "submit_as_job.py". <br>
If timestep data is saved, "calc_metric.py" can isolate the targeted calculation for quick plotting / printing. <br>


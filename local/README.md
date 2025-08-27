# local
<br>
<img src='logo.png' align="right" height="139" />
This repository includes scripts to generate all figures for the paper: "Large-Scale Clustering of Tropical Precipitation and its Implications for the Radiation Budget across Timescales" (excluding figures from the supporting information). <br>

### Repository structure
```bash
├── visualize/
│    └── doc_trop/
│        ├── changes_with_warming/...
│        │    └── map_projections/
│        │        └── regression_maps.py
│        └── interannual_variability/...
│            └── boxplots/...
│                └── correlation_partial.py
├── utils/
│   ├── util_calc/...
│   │       └── anomalies/
│   │           └── monthly_anomalies/
│   │               └── detrend_anom.py
│   ├── util_cmip/
│   │   └── model_letter_connection.py
│   └── user_specs.py
└── environment.yml
```

### How to use repository
First, change the paths in utils/user_specs.py such that the scripts know from where to access metrics and save figures. <br>
Next, change the working directory to where the "local" folder is. This is necessary as all scripts import the "utils" folder from the current working directory. <br>
Finally, run any script with the desired metrics and associated limits to generate figure. <br>






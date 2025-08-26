# Large-Scale Clustering of Tropical Precipitation and its Implications for the Radiation Budget across Timescales
<br>
<img src='logo.png' align="right" height="139" />
This Github repository include scripts that show how metrics and figures were generated for the paper: <br>
"Large-Scale Clustering of Tropical Precipitation and its Implications for the Radiation Budget across Timescales" <br>
DOI: '' <br>

<br>

**Authors** [name, affiliation, email, github username]  
[Philip Blackberg,      Monash University,              philip.blackberg@monash.edu,    [PBlackberg](https://github.com/PBlackberg?tab=repositories)] (corresponding)

<br>

### Repository structure
```bash
large-scale-clustering-of-tropical-precipitation/
├── LICENCE
├── README.md
├── gadi
│   ├── get_metrics/
│   │   ├── models/
│   │   │   └── cmip/..
│   │   │       └── doc_metrics/..
│   │   │           └── mean_area/
│   │   │               ├── calc_metric.py
│   │   │               ├── main_func.py
│   │   │               └── submit_as_job.py
│   │   ├── observations/..
│   │   │   └── GPCP/..
│   │   │       └── visualization/
│   │   │           ├── helper_funcs
│   │   │           ├── plots
│   │   │           ├── plots2
│   │   │           ├── calc_metric.py
│   │   │           ├── main_func.py
│   │   │           └── submit_as_job.py
│   └── utils/..
│       ├── util_calc/..
│       │       └── doc_metrics/..
│       │           └── mean_area/
│       │               └── mean_area.py
│       ├── util_cmip/..
│       │       ├── get_cmip_data.py
│       │       └── ecs_data.py
│       ├── util_obs/..
│       │       └── get_GPCP_data.py
│       ├── util_qsub/
│       │       └── interactive_script.py
│       │       └── submission_funcs.py
│       └── user_specs.py
├── local
    ├── utils
    │   ├── util_calc/
    │   │       └── anomalies/
    │   │           └── monthly_anomalies/
    │   │               └── detrend_anom.py
    │   ├── util_cmip/
    │   │       └── model_letter_connection.py
    │   └── user_specs.py
    └── visualize
        └── doc_trop
            ├── changes_with_warming/..
            │    └── map_projections/
            │        └── regression_maps.py
            └── interannual_variability/..
                └── boxplots/..
                    └── correlation_partial.py

```
Scripts to generate metrics are found in the "gadi" folder. <br>
These metrics can also be found on Zenodo: <br>
Link: <br>
DOI:  <br>
Scripts to generate figures are found in the "local" folder <br>

These scripts are just examples of how metrics were generated and how figures were created. A more complete repository is available upon request by email with corresponding author: <br>
philip.blackberg@monash.edu, carlphiliposcar@gmail.com <br>
















































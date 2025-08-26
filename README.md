# Large-Scale Clustering of Tropical Precipitation and its Implications for the Radiation Budget across Timescales
<br>
<img src='logo.png' align="right" height="139" />
This Github repository includes scripts that show how metrics and figures were generated for the paper: <br>
"Large-Scale Clustering of Tropical Precipitation and its Implications for the Radiation Budget across Timescales" <br>
DOI: '' <br>

<br>

**Authors** [name, affiliation, email, github username]  
[Philip Blackberg,      Monash University,              philip.blackberg@monash.edu,    [PBlackberg](https://github.com/PBlackberg?tab=repositories)] (corresponding)

**Abstract** <br>
The spatial organization of deep convection in tropical regions is posited to play an important role in determining characteristics of the tropical climate such as the humidity distribution and cloudiness and may therefore be an important control on climate feedbacks. This study analyzes one aspect of convective organization, the clustering of heavy precipitation on large scales, in both interannual variability and under warming in future climate projections. Both observations and global climate models indicate that large-scale clustering is sensitive to the SST gradient in the Pacific, being largest during El Ni\~no events. Under future warming, models project an increase in clustering with a large intermodel spread. The increase is associated with a narrowing of the intertropical convergence zone, while the model spread is partially explained by differences in projections of the SST gradient in the Pacific. Both observations and models indicate large-scale clustering influences the cloud and humidity distributions, albeit with some differences. However, the intermodel spread in changes in clustering with warming is not related to the intermodel spread in projections of tropical-mean relative humidity or low cloudiness in regions of descent, precluding attempts to provide an observational constraint on feedbacks or climate sensitivity. Nevertheless, the tendency for a meridional contraction of precipitation explains about 45\% of the variance in projected drying, highlighting the narrowing of the ITCZ as an important aspect of large-scale convective organization in a warmer climate.  

<br>

**Repository structuret** 
```bash
large-scale-clustering-of-tropical-precipitation/
├── LICENCE
├── README.md
├── gadi
│   ├── get_metrics/
│   │   ├── models/
│   │   │   └── cmip/...
│   │   │       └── doc_metrics/...
│   │   │           └── mean_area/
│   │   │               ├── calc_metric.py
│   │   │               ├── main_func.py
│   │   │               └── submit_as_job.py
│   │   ├── observations/...
│   │   │   └── GPCP/...
│   │   │       └── visualization/
│   │   │           ├── helper_funcs/
│   │   │               └── plot_func_map.py
│   │   │           ├── plots/...
│   │   │           │   └── conv_itcz_cl_0.png
│   │   │           ├── plots2/...
│   │   │           │   └── conv_itcz_cl_0.svg
│   │   │           ├── calc_metric.py
│   │   │           ├── main_func.py
│   │   │           └── submit_as_job.py
│   └── utils/...
│       ├── util_calc/...
│       │   └── doc_metrics/...
│       │       └── mean_area/
│       │           └── mean_area.py
│       ├── util_cmip/...
│       │   ├── get_cmip_data.py
│       │   └── ecs_data.py
│       ├── util_obs/...
│       │   └── get_GPCP_data.py
│       ├── util_qsub/
│       │   ├── interactive_script.py
│       │   └── submission_funcs.py
│       └── user_specs.py
└── local
    ├── utils
    │   ├── util_calc/...
    │   │       └── anomalies/
    │   │           └── monthly_anomalies/
    │   │               └── detrend_anom.py
    │   ├── util_cmip/
    │   │       └── model_letter_connection.py
    │   └── user_specs.py
    └── visualize
        └── doc_trop/
            ├── changes_with_warming/...
            │    └── map_projections/
            │        └── regression_maps.py
            └── interannual_variability/...
                └── boxplots/...
                    └── correlation_partial.py

```

**How to use repository** 
Scripts to generate metrics are found in the "gadi" folder. <br>
These metrics can also be found on Zenodo: <br>
Link: <br>
DOI:  <br>
Scripts to generate figures are found in the "local" folder <br>
This repository only includes examples of how the key metrics and figures were generated. A more complete repository is available upon request.




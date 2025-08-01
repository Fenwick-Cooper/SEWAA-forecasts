# SEWAA-forecasts

Scripts for operational rainfall forecasts in the ICPAC region.

## How to use

See the header of the master scripts

6h_accumulations/sixHourly_script.py

24h_accumulations/seven_day_script.py

## Requires

https://github.com/Fenwick-Cooper/ensemble-cgan/tree/Jurre_brishti_2

https://github.com/Fenwick-Cooper/ensemble-cgan/tree/Mvua_kubwa

## Roadmap

### Update to new cGAN

1. Implement scripts for new cGAN models.
2. Finalise working new models on github.
3. Place working data (weights, elevation etc online).

### Combine with HTML interface

1. Add HTML interface.
2. Update HTML interface with list of improvements.
3. Add ELR quality plots to HTML interface.

### Add ELR

1. Implement ELR scripts for ELR models.
2. ELR interface python prototype.
3. Add ELR to HTML interface.

### Prepare for remote install

1. Make all paths relative to base directory.
2. Download from gbmc instead of ECMWF
3. Remove upload to gbmc
4. Make prototype cGAN installation script for Windows/Linux/Mac.


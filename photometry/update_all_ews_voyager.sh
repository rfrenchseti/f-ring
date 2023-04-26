#!/usr/bin/bash
export BKGND_SUB_MOSAIC_DIR=/data/cb-results/fring/ring_mosaic/bkgnd_sub_mosaic_FRING_VOYAGER1
python create_ews.py --output-csv-filename ../data_files/v1_ew_0_1.csv --slice-size 1 --minimum-coverage 0 --minimum-slice-coverage 0.2 --maximum-slice-resolution 100
export BKGND_SUB_MOSAIC_DIR=/data/cb-results/fring/ring_mosaic/bkgnd_sub_mosaic_FRING_VOYAGER2
python create_ews.py --output-csv-filename ../data_files/v2_ew_0_1.csv --slice-size 1 --minimum-coverage 0 --minimum-slice-coverage 0.2 --maximum-slice-resolution 100

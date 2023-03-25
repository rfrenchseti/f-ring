#!/usr/bin/bash
python create_ews.py --output-csv-filename ../data_files/ew_stats_1zone_0_1.csv --slice-size 1 --minimum-coverage 0
#python create_ews.py --output-csv-filename ../data_files/ew_stats_1zone_60_10.csv --slice-size 10 --minimum-coverage 60
python create_ews.py --output-csv-filename ../data_files/ew_stats_1zone_60_0.csv --slice-size 0 --minimum-coverage 60

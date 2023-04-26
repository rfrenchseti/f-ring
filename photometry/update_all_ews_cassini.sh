#!/usr/bin/bash
python create_ews.py --output-csv-filename ../data_files/cass_ew_0_1.csv --slice-size 1 --minimum-coverage 0
python create_ews.py --output-csv-filename ../data_files/cass_ew_60_0.csv --slice-size 0 --minimum-coverage 60

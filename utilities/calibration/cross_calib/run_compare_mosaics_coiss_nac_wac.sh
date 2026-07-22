#!/bin/sh
# --max-pixel-incidence 60 --results-dir coiss_nacresults_lscap60
python run_compare_mosaics.py --set1-kind coiss --set2-kind coiss --set1-dir /seti/research/f-ring/calibration/coiss_nac_wac_reproj/nac --set2-dir /seti/research/f-ring/calibration/coiss_nac_wac_reproj/wac --matches coiss_nac_wac_calib/coiss_nac_wac_matches.csv $*


./run_compare_mosaics_coiss_nac_wac.sh --photometry lommel_seeliger --max-pixel-incidence 60 --results-dir results_lscap60 --results-dir coiss_nac_wac_calib/results_lscap60

#  9741  python run_compare_mosaics.py --photometry lommel_seeliger results_ls
#  9742  python run_compare_mosaics.py --photometry lommel_seeliger --results-dir results_ls
#  9744  python run_compare_mosaics.py --photometry lommel_seeliger --results-dir results_ls
#  9762  python run_compare_mosaics.py --photometry minnaert --results-dir results_min
#  9827  python run_compare_mosaics.py --photometry lommel_seeliger --max-pixel-incidence 70 --results-dir results_lscap70
#  9830  python run_compare_mosaics.py --photometry lommel_seeliger --max-pixel-incidence 60 --results-dir results_lscap60

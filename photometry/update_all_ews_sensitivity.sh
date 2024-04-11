#!/bin/bash
export BKGND_SUB_MOSAIC_DIR=/data/cb-results/fring/ring_mosaic/bkgnd_sub_mosaic_FMOVIE_SENSITIVITY

for RZOOM in 1 2 3 4 5 6 7 8 9 10; do
  for LZOOM in 1 5; do
    for RRES in 1 5; do
      for LRES in 0.02 0.10; do
        echo RR $RRES LR $LRES RZ $RZOOM LZ $LZOOM
        python create_ews.py --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES --output-csv-filename ../data_files/sensitivity/sens_ew_${RRES}_${LRES}_${RZOOM}_${LZOOM}_0_0.csv --minimum-coverage 0
      done
    done
  done
done

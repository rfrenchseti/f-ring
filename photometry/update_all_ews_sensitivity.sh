#!/usr/bin/bash
export BKGND_SUB_MOSAIC_DIR=/data/cb-results/fring/ring_mosaic/bkgnd_sub_mosaic_FMOVIE_SENSITIVITY
#!/usr/bin/bash
for RZOOM in 1 2 5 10; do
  for LZOOM in 1 2 5 10; do
    for RRES in 2 5 10 50; do
      for LRES in 0.02 0.10; do
        echo $RZOOM $LZOOM $RRES $LRES
          python create_ews.py --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES --output-csv-filename ../data_files/sensitivity/sens_ew_${RRES}_${LRES}_${RZOOM}_${LZOOM}_0_0.csv --minimum-coverage 0 SENS_ISS_181RF_FMOVIE001_PRIME SENS_ISS_181RF_FRINGPHOT001_VIMS SENS_ISS_196RF_FMOVIE006_PRIME
      done
    done
  done
done

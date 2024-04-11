#!/usr/bin/bash
cd ../mosaics

for RZOOM in 1 2 3 4 5 6 7 8 9 10; do
  for LZOOM in 1 5; do
    for RRES in 1 5; do
      for LRES in 0.02 0.10; do
        echo RR $RRES LR $LRES RZ $RZOOM LZ $LZOOM
        python ring_ui_reproject.py --max-subprocesses 3 --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES --verbose # --recompute-reproject
        python ring_ui_mosaic.py --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES # --recompute-mosaic
        python ring_ui_bkgnd.py --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES # --recompute-bkgnd
      done
    done
  done
done

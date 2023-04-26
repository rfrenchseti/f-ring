#!/usr/bin/bash
for RZOOM in 1 2 5 10; do
  for LZOOM in 1 2 5 10; do
    for RRES in 2 5 10 50; do
      for LRES in 0.02 0.10; do
        echo $RZOOM $LZOOM $RRES $LRES
        python ring_ui_reproject.py --max-subprocesses 3 --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES
        python ring_ui_mosaic.py --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES
        python ring_ui_bkgnd.py --ring-type FMOVIE_SENSITIVITY --radial-zoom $RZOOM --longitude-zoom $LZOOM --radius-resolution $RRES --longitude-resolution $LRES
      done
    done
  done
done

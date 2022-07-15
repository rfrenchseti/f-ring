sudo apt install python3.10-tk

Program list:

- compare_cisscal_versions.py
    Using the old and new .IMG files in ~/DS/f-ring/compare_cisscal_versions,
    figure out where the F ring is in the image and collect pixels in that
    general area, then print statistics about the mean and median of those
    pixels between old and new files.
    Goes with notebooks/anayze_cisscal_calibration.ipyn

- create_prometheus_dist.py
    Create a table of minimum distance between Prometheus and the F ring core
    for a series of dates or for the specific dates of the observations.
    


- f_ring_util.py
    General utility routines used by all programs.

- imgdisp.py
- julian.py
- julian_dateparser.py
- textkernel.py
    Copied from other locations.

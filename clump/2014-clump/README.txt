Author: Shannon Hicks
Date: 10/03/2012
Updated by Rob French 6/15/2013

This is a file detailing the functions of the major programs in this directory.
For more detailed explanations of each program see the header in their respective files.

MOSAIC PROGRAMS
===============

These programs are used to view and manipulate the Cassini mosaics.

find_eccentric_anomaly.py
ring_bkgnd.py
ring_ew.py
ring_mosaic.py
ring_offset_stats.py
ring_repro_to_png.py
ring_reproject.py
ring_toplevel.py
ringimage.py
ringutil.py
fitreproj.py
imgdisp.py
modelbkgnd.py
plot_ew_phase_curve.py
plot_az_profiles.py
mosaic.py

ringutil.py - houses all classes used by the gui system as well as other helper functions.

ring_toplevel.py allows you to view mosaics, individual images, recalculate reprojections, etc. It calls the other ring_* programs as well as imgdisp, fitreproj, and modelbkgnd.py.

plot_az_profiles.py plots EW profiles in various ways.  


CLUMP ANALYSIS PROGRAMS
=======================

clump_find_wavelet.py
clump_track.py

clumputil.py - Analagous to ringutil.py
Stores the ClumpData and ClumpDBEntry classes and various other clump helper functions. Must import these classes if you plan to use them.

To create a working clump database use the "clump_find_wavelet.py" program.
It can create three kinds of databases-full cassini database, downsampled cassini database, and voyager.

clump_correlate_attributes.py - Correlate height, width, slopes, etc. for all approved Cassini clumps and print out a correlation matrix.
[Verified and code reviewed OK RF 6/15/13]


CLUMP GUI
=========

clump_gui.py
clumpmapdisp.py

In order to use the clump GUI you need to run the clump_track.py on the full clump database created by clump_find_wavelet. 

The clump GUI allows us to visualize the tracking procedure by marking chains that satisfy whatever criteria was specified in the options.
Buttons and functions are stored in both clump_gui.py and clumpmapdisp.py.


CLUMP MANIPULATION PROGRAMS
===========================

clump_add_wavelet.py - Add a new clump to the master clump database based on a center and guessed width. This is for the case where
the wavelet system doesn't detect the clump we know is there.

clump_change_database_ews.py - Take the Cassini and Voyager approved lists and approved databases and replace all EWs and clump
brightness information with the 15th percentile EW. [Verified and code reviewed OK RF 6/15/13]

clump_longitude_error.py - Calculate mean motion errors, including with splitting clumps.
Read approved clumps list with lives and write approved clumps list with errors.
[Verified and code reviewed OK RF 6/15/13]

lifetimes_analysis.py - Calculate min and max lifetimes for each clump. Read approved clumps list and write
approved clumps list with lives list.

remove_clump_24.py


HELPER MODULES
==============

bandpass_filter.py - called by several programs to filter the EW profiles of both Cassini and Voyager data.

clump_gaussian_fit.py - Fit a Gaussian to a clump.

clump_radial_reproject.py

cwt.py

create_radial_mosaics.py - creates FULL radial mosaics of specified profiles.
It is different from clump_radial_reprojection because it is autonomous from the GUI and does NOT create zoomed in images.
It has the option to draw clumps.

shannon_animation.py - creates two movies (.mp4 and .avi) out of the radial mosaics produced by "create_radial_mosaics.py".
Note: This program requires mencoder and therefore only runs on a linux machine!


VOYAGER PROGRAMS
================

check_voyager_clumps.py - Dump clump longitudes and chains for voyager approved clumps.
[Verified and code reviewed OK RF 6/15/13]

voyager_analysis.py - creates the distribution histograms and statistical analysis of voyager clumps and profiles.
Requires a voyager database produced by clump_find_wavelet.py.


STATISTICS AND PLOTTING PROGRAMS
================================

clump_creation_rate_monte_carlo.py - Analyze clump creation rate over different time periods by doing
Monte Carlo simulation on available longitudes and observations.

clump_good_obsids.py - Find the good obsids for the union method of counting clumps.
[Verified and code reviewed OK RF 6/15/13]

clump_relations.py - Plot various clump relationships:
	Time vs. lifetime
	Size vs. lifetime
	Number of clumps vs. lifetime
	Distance from Prometheus vs. lifetime
	Change in width over time
	Change in brightness over time
	Brightness vs. true longitude
	Number of clumps vs. time
	Width vs. brightness scatter
	Clump longitude vs. prometheus histogram
	Delta W vs. brightness and Delta Brightness vs. Width
	
clump_table.py - Dump out a CSV file of clump information.
[Verified and code reviewed OK RF 6/15/13]

clump_viewing_geometries.py - Print out the emission/phase/incidence angles for every obs in every chain.

paper_figures.py

plot_az_profiles.py - Complicated options for plotting profiles

plot_clump_chains.py - Plot each chain with mosaics.

plot_ew_phase_curve.py - Plot mean EW by phase, incidence, emission, etc to experiment with relationships
and different baselines.

rf_experiments.py - Clump and chain validation.  Plot a series of OBS with approved clumps marked for
visual verification.

ring_compare_phase_curve.py - Compare our phase curve with Brightening paper.


ATTREE DATA ANALYSIS PROGRAMS
=============================

attree_corotating_coords.py - Convert the Attree mini-jet/feature spreadsheet into co-rotating coords and write a new table
[Verified longitude coordinate system, code reviewed OK RF 6/15/13]

monte_carlo_attree.py - Monte Carlo simulation to see how many Attree features we would expect to see with our
clump width distribution. Also various plots about Attree's database.




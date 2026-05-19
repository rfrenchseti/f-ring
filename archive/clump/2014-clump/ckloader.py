#!/usr/bin/python
################################################################################
# ckloader.py
#
# Mark R. Showalter, SETI Institute, July 2009
################################################################################

from cspice import *
import os, fnmatch

class CKloader():

    def __init__(self, dirpath, frame_id):
        """Constructor for a CK loader.

        Input:
            dirpath = path to a single directory containing all the CK files
                      required.

            frame_id = the SPICE ID for the coordinate frame needed, e.g.,
                      -82000 for Cassini.

        Return: A new CKloader object to keep track of C kernels and load them when
                needed.
        """

        self.dirpath = dirpath
        self.frame_id = frame_id

        self.ets = []
        self.loaded = []

        self.slop = spd() / 2.
        self.verbose = False

        # List all the files in the directory
        filenames = os.listdir(dirpath)

        # Select the C kernels
        filenames = fnmatch.filter(filenames, "*.bc")

        last_et = None
        # Tabulate time limits
        for f in filenames:
            longfile = os.path.join(dirpath,f)

            limits = ckcov(longfile, frame_id, False, "SEGMENT", 0., "TDB")[0]
            self.ets.append((limits[0], limits[1], f))

            self.loaded.append(False)
#            self.verbose = True
            if self.verbose:
                if last_et != None:
                    if limits[0] > last_et:
                        print 'Missing CK ETs between', last_et, 'and', limits[0], ' - next file', f, '(', limits[0], 'to', limits[1], ')'
                last_et = limits[1]

        self.ets.sort()

    # Make sure the CK is loaded for a particular ephemeris time
    def furnishET(self, et):
        """Method to load the C kernel(s) for a particular ephemeris time.

        Input:
            et = the ephemeris time (TDB) required.

        Side effects: The needed CK is loaded if necessary. In verbose mode, a
            message is printed.
        """

        self.furnishETs(et, et)

    # Make sure the CKs are loaded for a particular range of ephemeris times
    def furnishETs(self, et1, et2):
        """Method to load the C kernel(s) required for  a particular range of
        ephemeris times.

        Input:
            et1 = the earliest ephemeris time (TDB) required.
            et2 = the latest ephemeris time required.

        Side effects: The needed CK files are loaded if necessary. In verbose,
            mode a message is printed for each file.
        """

        # Allow for 12 hours of slop
        t1 = min(et1, et2) - spd()/2.
        t2 = max(et1, et2) + spd()/2.

        # Search list for CKs with overlapping times
        for i in range(len(self.ets)):
            if self.ets[i][1] < t1: continue
            if self.ets[i][0] > t2: continue

            # Eliminate anything already loaded
            if self.loaded[i]: continue

            # Load if needed
            f = self.ets[i][2]
            longfile = os.path.join(self.dirpath, f)

            furnsh(longfile)
            self.loaded[i] = True
            if self.verbose: print "Loading CK: " + f

    # Define the amount of time by which to expand time ranges when loading
    def setSlop(self, dt):
        """Method to control the amount of 'slop' to allow when loading C
        kernels.

        Input:
            dt = the number of seconds by which to expand the range of ephemeris
                 times to load, in seconds. Default = 43200, or 12 hours.
        """

        self.slop = dt

    # Turn verbose mode on or off
    def setVerbose(self, flag=True):
        """Method to control the verbosity of the CK loader. In verbose mode, a
        message is printed each time a new CK file is loaded.

        Input:
            flag = True to enter verbose mode, False to leave it. Initially
                   False.
        """

        self.verbose = flag

################################################################################
# Test program
################################################################################

def test():
        furnsh("/Users/doug/cspice/kernels/naif0009.tls")
        furnsh("/Users/doug/cspice/kernels/cas00112.tsc")
        ckloader = CKloader("/Users/doug/cspice/kernels/Cassini/CK-reconstructed/", -82000)
        ckloader.setVerbose()

        print "January 2008..."
        et1 = utc2et("2008-01-01")
        et2 = utc2et("2008-02-01")

        ckloader.furnishETs(et1, et2)

        print "February 2008..."
        et1 = et2
        et2 = utc2et("2008-03-01")

        ckloader.furnishETs(et1, et2)

        print "Days of March..."
        et = et2 - spd()

        for i in range(1,32):
            et = et + spd()

            print i
            ckloader.furnishET(et)

# Execute the main test progam if this is not imported
if __name__ == "__main__": test()


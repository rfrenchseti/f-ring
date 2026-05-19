#!/usr/bin/python
################################################################################
# spkloader.py
#
# Mark R. Showalter, SETI Institute, July 2009
################################################################################

from cspice import *
import os, fnmatch

class SPKloader():

    def __init__(self, dirpath, body_id):
        """Constructor for an SPK loader.

        Input:
            dirpath = path to a single directory containing the SPK files
                      required.

            frame_id = the SPICE ID for the body needed, e.g., -82 for Cassini.

        Return: A new SPKloader object to keep track of SP kernels and load them
                when needed.
        """

        self.dirpath = dirpath
        self.body_id = body_id

        self.ets = []
        self.loaded = []

        self.slop = spd() / 2.
        self.verbose = False

        # List all the files in the directory
        filenames = os.listdir(dirpath)

        # Select the C kernels
        filenames = fnmatch.filter(filenames, "*.bsp")

        last_et = None
        # Tabulate time limits
        for f in filenames:
            longfile = os.path.join(dirpath,f)

            limits = spkcov(longfile, body_id)[0]
            self.ets.append((limits[0], limits[1], f))

            self.loaded.append(False)
            if last_et != None:
                if limits[0] > last_et:
                    print 'Missing SPK ETs between', last_et, 'and', limits[0], ' - next file', f
            last_et = limits[1]
        self.ets.sort()

    # Make sure the SPK is loaded for a particular ephemeris time
    def furnishET(self, et):
        """Method to load the SP kernel(s) for a particular ephemeris time.

        Input:
            et = the ephemeris time (TDB) required.

        Side effects: The needed SPK is loaded if necessary. In verbose mode, a
            message is printed.
        """

        self.furnishETs(et, et)

    # Make sure the SPKs are loaded for a particular range of ephemeris times
    def furnishETs(self, et1, et2):
        """Method to load the SP kernel(s) required for  a particular range of
        ephemeris times.

        Input:
            et1 = the earliest ephemeris time (TDB) required.
            et2 = the latest ephemeris time required.

        Side effects: The needed SPK files are loaded if necessary. In verbose,
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
            if self.verbose: print "Loading SPK: " + f

    # Define the amount of time by which to expand time ranges when loading
    def setSlop(self, dt):
        """Method to control the amount of 'slop' to allow when loading SP
        kernels.

        Input:
            dt = the number of seconds by which to expand the range of ephemeris
                 times to load, in seconds. Default = 43200, or 12 hours.
        """

        self.slop = dt

    # Turn verbose mode on or off
    def setVerbose(self, flag=True):
        """Method to control the verbosity of the SPK loader. In verbose mode, a
        message is printed each time a new SPK file is loaded.

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
        spkloader = SPKloader("/Users/doug/cspice/kernels/Cassini/SPK-reconstructed/", -82)
        spkloader.setVerbose()

        print "January 2008..."
        et1 = utc2et("2008-01-01")
        et2 = utc2et("2008-02-01")

        spkloader.furnishETs(et1, et2)

        print "February 2008..."
        et1 = et2
        et2 = utc2et("2008-03-01")

        spkloader.furnishETs(et1, et2)

        print "Days of March..."
        et = et2 - spd()

        for i in range(1,32):
            et = et + spd()

            print i
            spkloader.furnishET(et)

# Execute the main test progam if this is not imported
if __name__ == "__main__": test()


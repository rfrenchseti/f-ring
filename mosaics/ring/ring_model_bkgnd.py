import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filter
import gc

def mask_image(image, cutoff_sigmas=7., ignore_fraction=0.01, row_blur=5,
               mask_fill_value=-999, debug=False):
    """Returns a copy of the image, with a mask obscuring the pixels believed
    to be invalid.

    Inputs:
    image           The original image as an un-masked array.

    cutoff_sigmas   In each iteration, pixels in a row with values more than
                    this number of standard deviations above the row's mean will
                    be masked.

    ignore_fraction In each iteration, pixels within this fraction of each row's
                    minimum and maximum values will be excluded from the
                    calculation of the mean and standard deviation.

    row_blur        The number of total adjacent rows in the same column that
                    also get masked when one row exceeds the threshold for
                    masking.

    Returns:
                    A new version of the array, with invalid pixels masked.
    """

    if debug:
        print('*** In mask_image')

    # View as a masked array
    view = image.view(ma.MaskedArray)

    # Mask at mask_fill_value - these are pixels that were never initialized
    # from a reprojected image and thus are invalid by definition
    view = ma.masked_equal(view, mask_fill_value)
    # Necessary because the mask starts as 'nomask' first, not an array of bools
    view.mask = ma.getmaskarray(view)
    if debug:
        # How many invalid pixels we started with
        initial_num_zeros = np.sum(view.mask)
        print('Initial num zeros', initial_num_zeros)
        print('Rows', view.shape[0], 'Columns', view.shape[1], end=' ')
        print('Total pixels', view.shape[0]*view.shape[1])
        print('Rows left active', np.sum(np.any(view.mask, axis=-1)))

    # Initially, all rows are active
    rows = np.ones((view.shape[0],), dtype="bool")
    # Track the statistics of each row
    masked_mean_all = ma.mean(view, axis=1)
    masked_sdev_all = ma.std( view, axis=1)

    # Iterate until no rows are active...
    while np.any(rows):
        if debug:
            print('Rows left', np.sum(rows))
        # Tabulate the extremes and mean of each row
        row_max = ma.max(view[rows], axis=1)
        row_min = ma.min(view[rows], axis=1)
        row_mean = ma.mean(view[rows], axis=1)

        # Narrow the extremes by the height_fraction
        mask_above = row_max - ignore_fraction * (row_max - row_mean)
        mask_below = row_min + ignore_fraction * (row_mean - row_min)

        # Temporarily mask all the pixels outside the range
        save_mask = view.mask[rows].copy()
        view.mask[rows] |= ((view.data[rows] >= mask_above[:,np.newaxis])
                        |   (view.data[rows] <= mask_below[:,np.newaxis]))

        # Calculate new statistics ignoring the excessive pixels
        masked_mean_all[rows] = ma.mean(view[rows], axis=1) # 1-D array
        masked_sdev_all[rows] = ma.std( view[rows], axis=1) # 1-D array

        # Locate excessive residuals in each active row (ignoring the really
        # excessive pixels)
        mask_above1 = masked_mean_all + cutoff_sigmas * masked_sdev_all # 1-D
        mask_below1 = masked_mean_all - cutoff_sigmas * masked_sdev_all # 1-D

        # Smoothing: each row gets the maximum/minimum of its neighbors
        # Since we mask above the maximum (below the minimum) this will
        # _reduce_ the amount of masking we do here
        mask_above = filter.maximum_filter1d(mask_above1, row_blur,
                                             mode='nearest')
        mask_below = filter.minimum_filter1d(mask_below1, row_blur,
                                             mode='nearest')

        # Find the pixels in each row that are out of bounds
        # (np.newaxis just does a transpose)
        deweight = (~save_mask & # Use the original mask
                    ((view.data[rows,:] >= mask_above[rows][:,np.newaxis]) |
                     (view.data[rows,:] <= mask_below[rows][:,np.newaxis])))

        # Mask out the pixels at these locations
        view.mask[rows] = save_mask | deweight

        # Update the list of active rows
        rows[rows] &= np.any(deweight, axis=-1)     # axis = 1 doesn't work

        # Print summary of changes for debugging
        if debug:
            print('Pixels changed', np.sum(deweight))
            print("Rows changed", np.sum(np.any(deweight, axis=-1)))
            print('RUNNING THROUGH AGAIN')

    if debug:
        print("Masked pixels =", np.sum(view.mask))
        print("Masked pixels (not counting zeros) =", np.sum(view.mask) - initial_num_zeros)
        print('Rows left active', np.any(rows, axis=-1))

    return view

def model_background(image, ring_rows=0.2, cutoff_sigmas=4, degree=2,
                     background_pixels=200, masked=False, debug=False,
                     debug_col=None):
    """Derives a linear/quadratic model for the bkgnd in each column of a mosaic.

    Input:
        image           a masked mosaic array.
        ring_rows       the rows of the mosaic occupied by the ring. Use a pair
                        of row indices, a pair of fractional row values between
                        zero and one, or a single fraction which is centered on
                        the middle row of the image.
        cutoff_sigmas   the number of sigmas off the mean, beyond which any
                        pixel in a column should be masked.
        degree          the order of the polynomial to fit: 1 or 2.
        background_pixels
                        the minimum number of valid pixels off the ring. If given as a
                        tuple, then the first number is the minimum inside the
                        ring and the latter is the minimum outside. If a single
                        integer is given it is the minimum number on each side.
                        Substitute floats to indicate the fractional number of
                        rows instead.
        masked          True to return a masked array in which the mask
                        identifies the pixels excluded from the polynomial fits;
                        False to return an unmasked array.
        debug           True to print out debugging information and progress
                        during the procedure.

    Return:             a masked array with the same dimensions as the mosaic.
                        The data array contains the model for every column that
                        it is available. The mask indicates which pixels were
                        used in the fit if input parameter masked=True.
    """
    if debug:
        print('*** In model_background')

    def fit_linear(array, columns, x):
        # Perform a linear regression on the un-masked pixels in each column
        #
        # We seek a function y = a x + b
        #
        # a = (N Sum[xy] - Sum[x] Sum[y]) / (N Sum[x^2] - Sum[x]^2)
        # b = (Sum[y] - a Sum[x]) / N

        x = x[:,np.newaxis]

        sum1   = ma.sum(array[:,columns] * 0 + 1   , axis=0) # N
        sumx   = ma.sum(array[:,columns] * 0 + x   , axis=0)
        sumx2  = ma.sum(array[:,columns] * 0 + x**2, axis=0)
        sumy   = ma.sum(array[:,columns]           , axis=0)
        sumxy  = ma.sum(array[:,columns] * x       , axis=0)

        a = (sum1 * sumxy - sumx * sumy) / (sum1 * sumx2 - sumx**2.)
        b = (sumy - a * sumx) / sum1

        del sum1
        sum1 = None
        del sumx
        sumx = None
        del sumx2
        sumx2 = None
        del sumy
        sumy = None
        del sumxy
        sumxy = None
        gc.collect()

        # Return model
        model = a*x + b

        del a
        a = None
        del b
        b = None
        gc.collect()

        return model

    def fit_quadratic(array, columns, x):

        # Note that ma.polyfit() does not seem to work properly. Hence the
        # quadratic fit procedure is written out explicitly here, in a way that
        # can operate on all columns simultaneously.
        #
        # Perform a quadratic regression on the un-masked pixels in each column
        #
        # We seek a function y = a x^2 + bx + c
        #
        # Solve:
        #   Sum[y]    = a N        + b Sum[x]   + c Sum[x^2]
        #   Sum[xy]   = a Sum[x]   + b Sum[x^2] + c Sum[x^3]
        #   Sum[x^2y] = a Sum[x^2] + b Sum[x^3] + c Sum[x^4]

        x = x[:,np.newaxis]

        sum1   = ma.sum(array[:,columns] * 0 + 1   , axis=0)
        sumx   = ma.sum(array[:,columns] * 0 + x   , axis=0)
        sumx2  = ma.sum(array[:,columns] * 0 + x**2, axis=0)
        sumx3  = ma.sum(array[:,columns] * 0 + x**3, axis=0)
        sumx4  = ma.sum(array[:,columns] * 0 + x**4, axis=0)
        sumy   = ma.sum(array[:,columns]           , axis=0)
        sumxy  = ma.sum(array[:,columns] * x       , axis=0)
        sumx2y = ma.sum(array[:,columns] * x**2    , axis=0)

        # Eliminate a:
        #   (N S[xy]   - S[x]S[y])   = b (N S[x^2] - S[x]S[x])
        #                            + c (N S[x^3] - S[x]S[x^2])
        #   (N S[x^2y] - S[x^2]S[y]) = b (N S[x^3] - S[x^2]S[x])
        #                            + c (N S[x^4] - S[x^2]S[x^2])

        y0 = sum1 * sumxy  - sumx  * sumy
        b0 = sum1 * sumx2  - sumx  * sumx
        c0 = sum1 * sumx3  - sumx  * sumx2

        y1 = sum1 * sumx2y - sumx2 * sumy
        b1 = sum1 * sumx3  - sumx2 * sumx
        c1 = sum1 * sumx4  - sumx2 * sumx2

        del sumx3
        sumx3 = None
        del sumx4
        sumx4 = None
        del sumxy
        sumxy = None
        del sumx2y
        sumx2y = None
        gc.collect()

        # Eliminate b:
        #   Y0 = b B0 + c C0
        #   Y1 = b B1 + c C1
        #
        #   (B0 Y1 - B1 Y0) = c (B0 C1 - B1 C0)

        c = (b0 * y1 - b1 * y0) / (b0 * c1 - b1 * c0)

        # Back-substitute for b

        b = (y0 - c * c0) / b0

        # Back-substitute for a

        a = (sumy - b * sumx - c * sumx2) / sum1

        del sum1
        sum1 = None
        del sumx
        sumx = None
        del sumx2
        sumx2 = None
        del sumy
        sumy = None
        gc.collect()

        # Return model
        model = a*x*x + b*x + c

        del a
        a = None
        del b
        b = None
        del c
        c = None
        gc.collect()

        return model

    # Create a "ringless" copy of the mosaic with the same data but the ring
    # masked out
    ringless = ma.array(image.data, copy=False) # We're not modifying the data, so just use the same array
    ringless.mask = ma.getmaskarray(image).copy() # But we are modifying the mask
#    print ring_rows
    #  Mask out the specified ring rows
    if ring_rows is not None:

        # A single int is the number of rows to mask, centered
        if type(ring_rows) == type(0):
            rmin = ringless.shape[0] - ring_rows/2
            rmax = ringless.shape[0] + ring_rows/2

        # A single float is the fractional height of the row to mask
        elif type(ring_rows) == type(0.):
            rmin = int(ringless.shape[0] * (1. - ring_rows)/2. + 0.5)
            rmax = int(ringless.shape[0] * (1. + ring_rows)/2. + 0.5)

        # Otherwise it had better be a tuple of ints
        else:
            rmin = ring_rows[0]
            rmax = ring_rows[1]

        ringless[rmin:rmax,:] = ma.masked
        center_row = (rmin + rmax) / 2
    else:
        center_row = image.shape[0]/2

    # Interpret the background pixel count
    # Interpret a tuple or list as the values for inside and outside
    if (isinstance(background_pixels, (list, tuple))):
        below = background_pixels[0]
        above = background_pixels[1]
    else:
        below = background_pixels
        above = background_pixels

    if isinstance(below, float):
        below = int(below * rmin + 0.5)

    if isinstance(above, float):
        above = int(above * (image.shape[0]-rmax) + 0.5)

    if debug:
        print('Ring min', rmin)
        print('Ring max', rmax)
        print('Ring center', center_row)
        print('Pixels above', above)
        print('Pixels below', below)

    # Define powers of x for polynomial evaluation
    x = np.arange(ringless.shape[0], dtype="float")
    x = 2. * (x - ringless.shape[0]/2.) / ringless.shape[0]  # Range from -1 to 1

    # Initialize the model and the tally of standard deviations
    model = ma.zeros(image.shape) # The polynomial model
    model.mask = ma.getmaskarray(model)                 # Make mask an array

    resid = ma.zeros(image.shape) # The residuals
    resid.mask = ma.getmaskarray(resid)                 # Make mask an array

    column_sigmas = ma.zeros(image.shape[1])
    column_sigmas.mask = ma.getmaskarray(column_sigmas) # Make mask an array

    # Initialize the set of active columns
    column_active = np.ones((image.shape[1],), dtype="bool")

    image_mask = ma.getmaskarray(image)
    print(rmin, rmax, image.shape)
    print(np.sum(~image_mask[0:rmin,:], axis=0))
    print(np.sum(~image_mask[rmax+1:, :], axis=0))
    print(below, above)
    reject = ((np.sum(~image_mask[0:rmin,:], axis=0) < below) |
              (np.sum(~image_mask[rmax+1:, :], axis=0) < above))

    if debug:
        print('Reject col', np.sum(reject), 'Debug col', debug_col, ': #MaskBelow', np.sum(~image_mask[0:rmin, debug_col], axis=0), end=' ')
        print('#MaskAbove', np.sum(~image_mask[rmax+1:, debug_col], axis=0), end=' ')
        print('Reject', reject[debug_col])

    ringless[:,reject] = ma.masked
    resid[:,reject] = ma.masked
    column_active[reject] = False
    column_sigmas[reject] = ma.masked

    # Iterate until no remaining columns are active...
    while np.any(column_active):

        # Fit the polynomial
        if degree == 1:
            model[:,column_active] = fit_linear(ringless, column_active, x)
        else:
            model[:,column_active] = fit_quadratic(ringless, column_active, x)
#        coeffts = ma.polyfit(x, ringless[:,columns], degree)
#        model = np.dot(xpowers, coeffts)

        resid[:,column_active] = (ringless[:,column_active] -
                                     model[:,column_active])**2.

        # Locate excessive residuals in each active column
        column_sigmas[column_active] = ma.std(resid[:,column_active], axis=0)
        overall_sigma = ma.median(column_sigmas)

        cutoff = cutoff_sigmas * overall_sigma
        deweight = (~ringless.mask[:,column_active] &
                    (np.abs(resid.data[:,column_active]) >= cutoff))

        # Mask out the pixels at these locations
        ringless.mask[:,column_active] |= deweight
        resid.mask[:,column_active] |= deweight

        # Update the list of active columns
        column_active[column_active] &= np.any(deweight, axis=0)

        # Print summary of changes for debugging
        if debug:
            print("Columns changed = ", np.sum(column_active))

        # Reject columns that now have too few unmasked pixels
        print(rmin, rmax)
        reject = ((np.sum(~image_mask[0:rmin,:], axis=0) < below) |
                  (np.sum(~image_mask[rmax+1: ,:], axis=0) < above))

        ringless.mask[:,reject] = True
        resid.mask[:,reject] = True
        column_sigmas.mask[reject] = True
        column_active[reject] = False

        del reject
        reject = None
        gc.collect()

    # Define the mask to return
    if masked:
        model.mask = ringless.mask

    # Always mask out the pixels masked in the mosaic, as well as all columns
    # that have too few unmasked pixels
    else:
        model.mask = image.mask
        model.mask[:,:] |= column_sigmas.mask[:]

    return model

def integrate_image(masked, model, drad=5.):
    """Returns the mosaic converted to a longitudinal profile.

    Input:
        masked      a masked mosaic, as returned by mask_image.
        model       a model mosaic, as returned by model_background.
        drad        the number of km per pixel along each column.

    Return:         a masked 1-D array containing the radial integral from each
                    valid column.
    """

    return ma.sum(masked - model, axis=0) * drad

################################################################################

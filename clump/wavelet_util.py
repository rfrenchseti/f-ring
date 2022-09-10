################################################################################
# wavelet_util.py
#
# General classes for creating wavelets and computing wavelet transforms.
################################################################################

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy.integrate import trapz


class MotherWavelet(object):
    """Parent class for all mother wavelets.

    Parameters
    ----------
    len_series : int
        Length of the series to be decomposed.

    scales : array or list
        Array of scales used to compute the CWT. These are in units of the sample
        frequency.

    sampf : float
        Sample frequency of the series to be decomposed. This is how many samples there
        are in a unit period.

    pad_to : int
        Pad the series to a total length `pad_to` using zero padding.

    normalize : bool
        If True, the normalized version of the mother wavelet will be used (i.e.
        the mother wavelet will have unit energy).

    fc : string
        Characteristic frequency type - use the 'bandpass' or 'center' frequency of
        the Fourier spectrum of the mother wavelet to relate scale to period.
    """

    def __init__(self, len_series, scales, sampf=1, pad_to=None, normalize=True,
                 fc='bandpass'):
        """Initialize a mother wavelet."""
        assert len_series is not None
        assert scales is not None

        self._len_series = len_series
        # Set total length of mother wavelet to account for zero padding
        if pad_to is None:
            self._len_wavelet = len_series
        else:
            self._len_wavelet = pad_to
        self._scales = np.asfarray(scales)
        self._sampf = sampf
        self._normalize = normalize
        self._fc = fc

        self._wavelet_vals = None
        self._coi_coefs = None
        self._coi = None
        self._coi_mask = None


    def get_wavelet_vals(self):
        """Get the values of the mother wavelet for all scales."""
        raise NotImplementedError

    def get_coi_coef(sampf):
        """XXX

        To follow the convention in the literature, please define your
        COI coef as a function of period, not scale - this will ensure
        compatibility with the scalogram method.
        """
        raise NotImplementedError

    def get_coi(self):
        """Compute cone of influence."""

        if self._coi is None:
            y1 =  self.coi_coef * np.arange(0, self._len_series / 2)
            y2 = -self.coi_coef * np.arange(0, self._len_series / 2) + y1[-1]
            self._coi = np.r_[y1, y2]
        return self._coi

    def get_mask(self):
        """Get mask for cone of influence."""
        if self._coi_mask is None:
            mask = np.ones(self._wavelet_vals.shape, dtype=np.bool)
            masks = self._coi_coef * self._scales
            for s in range(len(self._scales)):
                if s != 0 and int(np.ceil(masks[s])) < mask.shape[1]:
                    mask[s, np.ceil(int(masks[s])):-np.ceil(int(masks[s]))] = False
            self._coi_mask = mask
        return self._coi_mask

    def cwt(self, x, weighting_function=lambda x: x**(-0.5), deep_copy=True):
        """Compute the continuous wavelet transform of x using an FFT.

        The cwt is defined as:

            T(a,b) = w(a) integral(-inf,inf)(x(t) * psi*{(t-b)/a} dt

        which is a convolution. In this algorithm, the convolution in the time
        (or space) domain is implemented as a multiplication in the Fourier domain.

        Parameters
        ----------
        x : 1D array
            Series to be transformed by the cwt

        weighting_function : Function used to weight the values
            Typically w(a) = a**(-0.5) is chosen as it ensures that the
            wavelets at every scale have the same energy.

        deep_copy : bool
            If true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.mother_wavelet; if false, wavelet.mother_wavelet will be a
            reference to the mother_wavelet object (that is, if you change the
            mother wavelet object, you will see the changes when accessing the
            mother wavelet through the wavelet object - this is NOT good for
            tracking how the wavelet transform was computed, but setting
            deep_copy to False will save memory).

        Returns
        -------
        Returns an instance of the CWTResult class. The values of the transform
        can be obtained using result.wt. Note that because this uses an FFT, the input
        data is assumed to be cyclical and there are no edge effects.

        Examples
        --------
        Create an instance of the SDG mother wavelet, normalized, using 10 scales and
        the center frequency of the Fourier transform as the characteristic frequency.
        Then, perform the continuous wavelet transform and plot the scalogram.

        import numpy as np
        from wavelet import SDGWavelet
        sampf = 64.
        x = np.arange(0, 2*np.pi, 2*np.pi/sampf)
        data = np.sin(x*2) + 0.5*np.sin(x*5)
        scales = np.arange(1, 20)
        sdg = SDGWavelet(len(data), scales, sampf=sampf, normalize=True, fc='bandpass')
        wavelet = sdg.cwt(data)
        wavelet.scalogram()

        References
        ----------
        Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook. Taylor
          and Francis Group, New York/London. 353 pp.
        """

        assert len(x) == self._len_series

        signal_dtype = x.dtype

        if len(x) < self._len_wavelet:
            # Zero pad as necessary
            n = len(x)
            x = np.resize(x, (self._len_wavelet,))
            x[n:] = 0

        # Transform the signal and mother wavelet into the Fourier domain
        xf = fft(x)
        mwf = fft(self._wavelet_vals.conj(), axis=1)

        # Convolve (multiply in Fourier space)
        wt_tmp = ifft(mwf * xf[np.newaxis,:], axis=1)

        # Shift output from ifft and multiply by weighting function
        wt = fftshift(wt_tmp, axes=[1]) * weighting_function(self._scales[:, np.newaxis])

        # If mother wavelet and signal are real, only keep real part of transform
        if np.all(np.isreal(self._wavelet_vals)) and np.all(np.isreal(x)):
            wt = wt.real

        return CWTResult(wt, self, weighting_function, signal_dtype, deep_copy)

    def cwt_nonfft(self, x, weighting_function=lambda x: x**(-0.5), deep_copy=True,
                   startx=None, endx=None):
        """Compute the continuous wavelet transform of x without using an FFT.

        The cwt is defined as:

            T(a,b) = w(a) integral(-inf,inf)(x(t) * psi*{(t-b)/a} dt

        which is a convolution. In this implementation we do not use an FFT to speed
        up the work for us.

        Parameters
        ----------
        x : 1D array
            Series to be transformed by the cwt

        weighting_function:  Function used to weight
            Typically w(a) = a^(-0.5) is chosen as it ensures that the
            wavelets at every scale have the same energy.

        deep_copy : bool
            If true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.mother_wavelet; if false, wavelet.mother_wavelet will be a
            reference to the mother_wavelet object (that is, if you change the
            mother wavelet object, you will see the changes when accessing the
            mother wavelet through the wavelet object - this is NOT good for
            tracking how the wavelet transform was computed, but setting
            deep_copy to False will save memory).

        Returns
        -------
        Returns an instance of the CWTResult class. The values of the transform
        can be obtained using result.wt. Note that because this is computed using
        direct correlation, the signal is not assumed to be cyclical and there are
        major edge effects.

        References
        ----------
        Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook. Taylor
          and Francis Group, New York/London. 353 pp.
        """

        assert len(x) == self._len_series

        signal_dtype = x.dtype

        if len(x) < self._len_wavelet:
            # Zero pad as necessary
            n = len(x)
            x = np.resize(x, (self._len_wavelet,))
            x[n:] = 0

        wt = np.zeros((len(self._scales), len(x)))

        if startx is None:
            startx = 0
        if endx is None:
            endx = len(x)

        startx = int(startx)
        endx = int(endx)

        for pos in range(startx, endx):
            if (pos % 100) == 0:
                print(pos, '/', len(x))
            x_idx_min = max(int(pos-self._wavelet_vals.shape[1]/2), 0)
            scale_idx_min = max(int(self._wavelet_vals.shape[1]/2-pos), 0)
            scale_idx_max = min(x_idx_min+self._wavelet_vals.shape[1], len(x))-x_idx_min
            scale_idx_len = scale_idx_max - scale_idx_min
            wt[:, pos] = np.sum(self._wavelet_vals[:, scale_idx_min:scale_idx_max] *
                                x[x_idx_min:x_idx_min+scale_idx_max-scale_idx_min],
                                axis=1)
            wt[:, pos] *= weighting_function(self._scales)

        # If mother wavelet and signal are real, only keep real part of transform
        if np.all(np.isreal(self._wavelet_vals)) and np.all(np.isreal(x)):
            wt = wt.real

        return CWTResult(wt, self, weighting_function, signal_dtype, deep_copy)

    def ccwt(self, x1, x2):
        """Compute the continuous cross-wavelet transform of 'x1' and 'x2'.

        Parameters
        ----------
        x1,x2 : 1D array
            Series used to compute cross-wavelet transform

        Returns
        -------
        Returns an instance of the Wavelet class.
        """

        return self.cwt(x1) * np.conjugate(self.cwt(x2))

    def icwt(self, wavelet):
        """Compute the inverse continuous wavelet transform.

        Parameters
        ----------
        wavelet : Instance of the Wavelet class

        Examples
        --------
        Use the Morlet mother wavelet to perform wavelet transform on 'data', then
        use icwt to compute the inverse wavelet transform to come up with an estimate
        of data ('data2'). Note that data2 is not exactly equal data.

        import matplotlib.pyplot as plt
        from scipy.signal import SDG, Morlet, cwt, icwt, fft, ifft
        import numpy as np

        x = np.arange(0,2*np.pi,np.pi/64)
        data = np.sin(8*x)
        scales = np.arange(0.5,17)

        mother_wavelet = MorletWavelet(len_series=len(data), scales=scales)
        wave_coefs=cwt(data, mother_wavelet)
        data2 = icwt(wave_coefs)

        plt.plot(data)
        plt.plot(data2)
        plt.show()

        References
        ----------
        Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook. Taylor
          and Francis Group, New York/London. 353 pp.
        """
        # If original wavelet was created using padding, make sure to include
        # information that is missing after truncation
        # (see self._wavelet_vals under __init__ in class CWTResult)
        if wavelet.mother_wavelet._len_series != wavelet.mother_wavelet._len_wavelet:
            full_wc = np.c_[wavelet.wt, wavelet.pad_vals]
        else:
            full_wc = wavelet.wt

        # Get wavelet values and take FFT
        wcf = fft(full_wc, axis=1)

        # Get mother wavelet values and take FFT
        mwf = fft(wavelet.mother_wavelet._wavelet_vals, axis=1)

        # Perform inverse continuous wavelet transform and make sure the result is the
        # same type (real or complex) as the original data used in the transform
        x = (1. / wavelet.mother_wavelet._cg *
             trapz(fftshift(ifft(wcf * mwf, axis=1), axes=[1]) /
                   (wavelet.mother_wavelet._scales[:,np.newaxis]**2),
                   dx = 1. / wavelet.mother_wavelet._sampf, axis=0))

        return x[0:wavelet.mother_wavelet._len_series].astype(wavelet.signal_dtype)


class SDGWavelet(MotherWavelet):
    """SDG Wavelet - second derivative of the Gaussian."""

    def __init__(self, *args, **kwargs):
        """Initialize the SDG mother wavelet."""

        super().__init__(*args, **kwargs)
        self._name = 'Second degree of a Gaussian (Mexican Hat)'

        # Set admissibility constant
        if self._normalize:
            self._cg = 4 * np.sqrt(np.pi) / 3.
        else:
            # Addison p. 9
            self._cg = np.pi

        # Define characteristic frequency
        # Addison p. 8
        if self._fc == 'bandpass':
            self._fc = np.sqrt(5./2.) * self._sampf / (2 * np.pi)
        elif self._fc == 'center':
            self._fc = np.sqrt(2.) * self._sampf / (2 * np.pi)
        else:
            raise ValueError(f'fc "{fc}" unknown')

        # coi_coef defined under the assumption that period is used, not scale
        # XXX
        self._coi_coef = 2 * np.pi * np.sqrt(2. / 5.) * self._fc # Torrence and
                                                                 # Compo 1998

        # Compute coefficients for the dilated mother wavelet
        self._wavelet_vals = self.get_wavelet_vals()

    def get_wavelet_vals(self):
        """Calculate the values for the SDG mother wavelet"""

        if self._wavelet_vals is None:
            # Create array containing values used to evaluate the wavelet function
            omega = np.arange(-self._len_wavelet / 2., self._len_wavelet / 2.)

            # Derivative of the Gaussian is:  (T&C 1998 p. 65)
            #   (-1)**(m+1) / sqrt(gamma(m+1/2)) * d**m/dm[exp(-xi**2/2)]
            # Normalized to unit energy:
            #   (-i)**m / sqrt(gamma(m+1/2)) (omega/s)**m exp[-(omega/s)**2/2]
            # Second derivative:
            #   (-1)**3 == -1
            #   gamma(2.5) = 3/4 * sqrt(pi)
            #   -1 / sqrt(3/4 * sqrt(pi)) (omega/s)**2 exp[-(omega/s)**2/2]

            # (omega/s)**2 = xi**2 / scale**2
            xsd2 = (omega / self._scales[:,np.newaxis]) ** 2

            if self._normalize:
                # 1 / sqrt(3/4 * sqrt(pi)) ==
                # 1 / sqrt(3/4) / pi**(1/4) ==
                # 1 / (sqrt(3) / sqrt(4)) / pi**(1/4) ==
                # 2 / sqrt(3) / pi**(1/4)
                c = 2. / np.sqrt(3) / np.power(np.pi, 0.25)
            else:
                c = 1.

            # Second Derivative Gaussian
            self._wavelet_vals = -c * xsd2 * np.exp(-xsd2 / 2.)

        return self._wavelet_vals

class FDG(MotherWavelet):
    # XXX THIS NEEDS TO BE UPDATED - DO NOT USE
    """Class for the FDG MotherWavelet (a subclass of MotherWavelet).

    NOTE: THIS IS THE SAME EXACT CLASS AS THE SDG EXCEPT FOR THE COEFFICIENTS

    SDG(self, len_series=None, pad_to=None, scales=None, sampf = 1,
        normalize=True, fc='bandpass')

    Parameters
    ----------
    len_series : int
        Length of time series to be decomposed.

    pad_to : int
        Pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set). This is used in the fft function when
        performing the convolution of the wavelet and mother wavelet in Fourier
        space.

    scales : array
        Array of scales used to initialize the mother wavelet.

    sampf : float
        Sample frequency of the time series to be decomposed.

    normalize : bool
        If True, the normalized version of the mother wavelet will be used (i.e.
        the mother wavelet will have unit energy).

    fc : string
        Characteristic frequency - use the 'bandpass' or 'center' frequency of
        the Fourier spectrum of the mother wavelet to relate scale to period
        (default == 'bandpass').

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt and
    icwt functions.

    Examples
    --------
    Create instance of FDG mother wavelet, normalized, using 10 scales and the
    center frequency of the Fourier transform as the characteristic frequency.
    Then, perform the continuous wavelet transform and plot the scalogram.

    x = np.arange(0, 2*np.pi, np.pi/8.)
    data = np.sin(x**2)
    scales = np.arange(10)

    mother_wavelet = FDG(len_series=len(data), scales=np.arange(10),
                         normalize=True, fc='center')
    wavelet = cwt(data, mother_wavelet)
    wave_coefs.scalogram()

    Notes
    -----
    None

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook. Taylor
      and Francis Group, New York/London. 353 pp.
    """

    def __init__(self,len_series=None,pad_to=None,scales=None,sampf=1,normalize=True, fc = 'bandpass'):
        """Initilize SDG mother wavelet"""

        self._name = 'fourth degree of a Gaussian'
        self._sampf = sampf
        self._scales = scales
        self._len_series = len_series
        self._normalize = normalize

        #set total length of wavelet to account for zero padding
        if pad_to is None:
            self._len_wavelet = len_series
        else:
            self._len_wavelet = pad_to

        #set admissibility constant
        if normalize:
            self.cg = 4 * np.sqrt(np.pi) / 3.
        else:
            self.cg = np.pi

        #define characteristic frequency
        if fc == 'bandpass':
            self._fc = np.sqrt(5./2.) * self._sampf / (2 * np.pi)
        elif fc == 'center':
            self._fc = np.sqrt(2.) * self._sampf / (2 * np.pi)
        else:
            raise CharacteristicFrequencyError("fc = %s not defined"%(fc,))

        # coi_coef defined under the assumption that period is used, not scale
        self.coi_coef = 2 * np.pi * np.sqrt(2. / 5.) * self._fc # Torrence and
                                                               # Compo 1998

        # compute coefficients for the dilated mother wavelet
        self._wavelet_vals = self.get_wavelet_vals()

    def get_wavelet_vals(self):
        """Calculate the coefficients for the SDG mother wavelet"""

        # Create array containing values used to evaluate the wavelet function
        xi = np.arange(-self._len_wavelet / 2., self._len_wavelet / 2.)

        # find mother wavelet coefficients at each scale
        xsd2 = -xi * xi / (self._scales[:,np.newaxis] * self._scales[:,np.newaxis])

        if self._normalize is True:
            c = 2. / (np.sqrt(3) * np.power(np.pi, 0.25))
        else:
            c = 1.

        mw = c*(xsd2*(xsd2 + 6) + 3.)*np.exp(xsd2/2.)   #Fourth Derivative Gaussian

        self._wavelet_vals = mw

        return mw



class Morlet(MotherWavelet):
    # XXX THIS NEEDS TO BE UPDATED - DO NOT USE
    """Class for the Morlet MotherWavelet (a subclass of MotherWavelet).

    Morlet(self, len_series = None, pad_to = None, scales = None,
           sampf = 1, f0 = 0.849)

    Parameters
    ----------
    len_series : int
        Length of time series to be decomposed.

    pad_to : int
        Pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set). This is used in the fft function when
        performing the convolution of the wavelet and mother wavelet in Fourier
        space.

    scales : array
        Array of scales used to initialize the mother wavelet.

    sampf : float
        Sample frequency of the time series to be decomposed.

    f0 : float
        Central frequency of the Morlet mother wavelet. The Fourier spectrum of
        the Morlet wavelet appears as a Gaussian centered on f0. f0 defaults
        to a value of 0.849 (the angular frequency would be ~5.336).

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt
    and icwt functions.

    Examples
    --------
    Create instance of Morlet mother wavelet using 10 scales, perform the
    continuous wavelet transform, and plot the resulting scalogram.

    # x = np.arange(0,2*np.pi,np.pi/8.)
    # data = np.sin(x**2)
    # scales = np.arange(10)
    #
    # mother_wavelet = Morlet(len_series=len(data), scales = np.arange(10))
    # wavelet = cwt(data, mother_wavelet)
    # wave_coefs.scalogram()

    Notes
    -----
    * Morlet wavelet is defined as having unit energy, so the `normalize` flag
      will always be set to True.

    * The Morlet wavelet will always use f0 as it's characteristic frequency, so
      fc is set as f0.

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook. Taylor
      and Francis Group, New York/London. 353 pp.

    """

    def __init__(self, len_series=None, pad_to=None, scales=None, sampf=1,
                 normalize=True, f0=0.849):
        """Initialize Morlet mother wavelet."""

        self._sampf = sampf
        self._scales = scales
        self._len_series = len_series
        self._normalize = True
        self._name = 'Morlet'

        # set total length of wavelet to account for zero padding
        if pad_to is None:
            self._len_wavelet = len_series
        else:
            self._len_wavelet = pad_to

        # define characteristic frequency
        self._fc = f0

        # Cone of influence coefficient
        self.coi_coef = 2. * self._sampf / (self._fc + np.sqrt(2. + self._fc**2) *
                        np.sqrt(2)); #Torrence and Compo 1998 (in code)

        # set admissibility constant
        # based on the simplified Morlet wavelet energy spectrum
        # in Addison (2002), eqn (2.39) - should be ok for f0 >0.84
        f = np.arange(0.001, 50, 0.001)
        y = 2. * np.sqrt(np.pi) * np.exp(-np.power((2. * np.pi * f -
            2. * np.pi * self._fc), 2))
        self.cg =  trapz(y[1:] / f[1:]) * (f[1]-f[0])

        # compute coefficients for the dilated mother wavelet
        self._wavelet_vals = self.get_wavelet_vals()

    def get_wavelet_vals(self):
        """Calculate the coefficients for the Morlet mother wavelet."""

        # Create array containing values used to evaluate the wavelet function
        xi=np.arange(-self._len_wavelet / 2., self._len_wavelet / 2.)

        # find mother wavelet coefficients at each scale
        xsd = xi / (self._scales[:,np.newaxis])

        mw = (np.power(np.pi,-0.25) *
              (np.exp(np.complex(1j) * 2. * np.pi * self._fc * xsd) -
               np.exp(-np.power((2. * np.pi * self._fc), 2) / 2.)) *
              np.exp(-np.power(xsd, 2) / 2.))

        self._wavelet_vals = mw

        return mw


class CWTResult(object):
    """Container for the results of a continuous wavelet transform."""

    def __init__(self, wt, mother_wavelet, weighting_function, signal_dtype,
                 deep_copy=True):
        """Initialization of CWTResult object.

        Parameters
        ----------
        wt : array
            Array of wavelet values.

        mother_wavelet : object
            Mother wavelet object used in the creation of `wt`.

        weighting_function : function
            Function used in the creation of `wt`.

        signal_dtype : dtype
            dtype of signal used in the creation of `wt`.

        deep_copy : bool
            If true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.mother_wavelet; if false, wavelet.mother_wavelet will be a
            reference to the mother_wavelet object (that is, if you change the
            mother wavelet object, you will see the changes when accessing the
            mother wavelet through the wavelet object - this is NOT good for
            tracking how the wavelet transform was computed, but setting
            deep_copy to False will save memory).

        Returns
        -------
        Returns an instance of the CWTResult class.
        """

        self.wt = wt[:, :mother_wavelet._len_series]

        # Store the values used for padding, if applicable
        if mother_wavelet._len_series != mother_wavelet._len_wavelet:
            self.pad_vals = wt[:, mother_wavelet._len_series:]
        else:
            self.pad_vals = None

        if deep_copy:
            self.mother_wavelet = deepcopy(mother_wavelet)
        else:
            self.mother_wavelet = mother_wavelet

        self.weighting_function = weighting_function
        self.signal_dtype = signal_dtype

    def get_gws(self):
        """Calculate Global Wavelet Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis. Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.
        """

        return self.get_wavelet_var()


    def get_wes(self):
        """Calculate Wavelet Energy Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis. Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.
        """

        coef = 1. / (self.mother_wavelet._fc * self.mother_wavelet._cg)

        return coef * trapz(np.abs(self.wavelet_vals)**2, axis=1)

    def get_wps(self):
        """Calculate Wavelet Power Spectrum."""

        # T&C p. 72 eq(22)
        return np.sqrt(np.sum(self.wt**2, axis=1) / len(self.wt))

        # return =  1. / self.mother_wavelet._len_series * self.get_wes()

    def get_wavelet_var(self):
        """Calculate Wavelet Variance (a.k.a. the Global Wavelet Spectrum of
        Torrence and Compo (1998)).

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis. Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.
        """

        coef = self.mother_wavelet._cg * self.mother_wavelet._fc

        wvar = coef / self.mother_wavelet._len_series * self.get_wes()

        return wvar

    def scalogram(self, show_coi=False, show_wps=False, ts=None, time=None,
                  use_period=True, ylog_base=None, xlog_base=None,
                  origin='top', figname=None):
        """Plot a scalogram.

        Creates a simple scalogram, with optional wavelet power spectrum and
        time series plots of the transformed signal.

        Parameters
        ----------
        show_coi : bool
            Set to True to see Cone of Influence

        show_wps : bool
            Set to True to see the Wavelet Power Spectrum

        ts : array
            1D array containing time series data used in wavelet transform. If set,
            time series will be plotted.

        time : array of datetime objects
            1D array containing time information

        use_period : bool
            Set to True to see figures use period instead of scale

        ylog_base : float
            If a log scale is desired, set `ylog_base` as float. (for log 10, set
            ylog_base = 10)

        xlog_base : float
            If a log scale is desired, set `xlog_base` as float. (for log 10, set
            xlog_base = 10) *note that this option is only valid for the wavelet power
            spectrum figure.

        origin : 'top' or 'bottom'
            Set origin of scale axis to top or bottom of figure

        Returns
        -------
        None

        Examples
        --------
        Create instance of SDG mother wavelet, normalized, using 10 scales and the
        center frequency of the Fourier transform as the characteristic frequency.
        Then, perform the continuous wavelet transform and plot the scalogram.

        x = np.arange(0,2*np.pi,np.pi/8.)
        data = np.sin(x**2)
        scales = np.arange(10)

        mother_wavelet = SDG(len_series=len(data), scales=np.arange(10),
                             normalize=True, fc='center')
        wavelet = cwt(data, mother_wavelet)
        wave_coefs.scalogram(origin = 'bottom')
        """

        sampf = self.mother_wavelet._sampf

        if ts is not None:
            show_ts = True
        else:
            show_ts = False

        if not show_wps and not show_ts:
            # only show scalogram
            figrow = 1
            figcol = 1
        elif show_wps and not show_ts:
            # show scalogram and wps
            figrow = 1
            figcol = 4
        elif not show_wps and show_ts:
            # show scalogram and ts
            figrow = 2
            figcol = 1
        else:
            # show scalogram, wps, and ts
            figrow = 2
            figcol = 4

        if time is None:
            x = np.arange(self.mother_wavelet._len_series) / sampf
        else:
            x = time

        if use_period:
            y = self.mother_wavelet._scales / self.mother_wavelet._fc
        else:
            y = self.mother_wavelet._scales * .04 * 2 #XXX

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(figrow, figcol, 1)

        # if show wps, give 3/4 space to scalogram, 1/4 to wps
        if show_wps:
            # create temp axis at 3 or 4 col of row 1
            axt = fig.add_subplot(figrow, figcol, 3)
            # get location of axtmp and ax1
            axt_pos = axt.get_position()
            ax1_pos = ax1.get_position()
            axt_points = axt_pos.get_points()
            ax1_points = ax1_pos.get_points()
            # set axt_pos left bound to that of ax1
            axt_points[0][0] = ax1_points[0][0]
            ax1.set_position(axt_pos)
            fig.delaxes(axt)

        if show_coi:
            # coi_coef is defined using the assumption that you are using
            #   period, not scale, in plotting - this handles that behavior
            if use_period:
                coi = self.mother_wavelet.get_coi() / self.mother_wavelet._fc / sampf
            else:
                coi = self.mother_wavelet.get_coi()

            coi[coi == 0] = y.min() - 0.1 * y.min()
            xs, ys = poly_between(np.arange(0, len(coi)), np.max(y), coi)
            ax1.fill(np.arange(len(coi)+1), np.append(coi, 0), 'k', alpha=0.4, zorder = 2)

        contf = ax1.contourf(x, y, np.abs(self.wt)**2, 100)
#        fig.colorbar(contf, ax=ax1, orientation='vertical', format='%2.1f')

        if ylog_base is not None:
            ax1.axes.set_yscale('log', basey=ylog_base)

        if origin == 'top':
            ax1.set_ylim((y[-1], y[0]))
        elif origin == 'bottom':
            ax1.set_ylim((y[0], y[-1]))
        else:
            raise OriginError('`origin` must be set to "top" or "bottom"')

        ax1.set_xlim((x[0], x[-1]))
        ax1.set_title('Scalogram')
        ax1.set_ylabel('Series')
        if use_period:
            ax1.set_ylabel('period')
            ax1.set_xlabel('time')
        else:
            ax1.set_ylabel('scales')
            if time is not None:
                ax1.set_xlabel('time')
            else:
                ax1.set_xlabel('sample')

        if show_wps:
            ax2 = fig.add_subplot(figrow, figcol, 4, sharey=ax1)
            if use_period:
                ax2.plot(self.get_wps(), y, 'k')
            else:
                ax2.plot(self.mother_wavelet._fc * self.get_wps(), y, 'k')

            if ylog_base is not None:
                ax2.axes.set_yscale('log', basey=ylog_base)
            if xlog_base is not None:
                ax2.axes.set_xscale('log', basey=xlog_base)
            if origin == 'top':
                ax2.set_ylim((y[-1], y[0]))
            else:
                ax2.set_ylim((y[0], y[-1]))
            if use_period:
                ax2.set_ylabel('period')
            else:
                ax2.set_ylabel('scales')
            ax2.grid()
            ax2.set_title('wavelet power spectrum')

        if show_ts:
            ax3 = fig.add_subplot(figrow, 2, 3, sharex=ax1)
            ax3.plot(x, ts)
            ax3.set_xlim((x[0], x[-1]))
#            ax3.legend(['time series'])
            ax3.grid()
            # align time series fig with scalogram fig
            t = ax3.get_position()
            ax3pos = t.get_points()
            ax3pos[1][0] = ax1.get_position().get_points()[1][0]
            t.set_points(ax3pos)
            ax3.set_position(t)
            if time is not None or use_period:
                ax3.set_xlabel('time')
            else:
                ax3.set_xlabel('sample')

        if figname is not None:
            plt.savefig(figname)
            plt.close('all')

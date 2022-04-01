##########################################################################################
# clump_bandpass_filter.py
#
# Bandpass filter for the EW profiles. This eliminates all but the mid-range frequencies,
# including Prometheus' 3.2 degree perturbance. Filtering is done in the frequency
# domain using FFTs.
#
# Note that 3.2 is the PERIOD of the oscillation. The actual FREQUENCY is 112.5 (360/3.2).
# The resulting profile is everything EXCEPT Prometheus' interference.
##########################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def fft_filter(ew_data, long_res_deg, obs_id, plot=False,
               hpf_period=30, lpf_period=90):
    # hpf_period and lpf_period are in cycles per complete ring

    if False:  # Use to inject a fake signal for testing
        longitudes = np.arange(ew_data.size) * long_res_deg
        ew_data = (np.sin(np.radians(longitudes*360/30)) +   # Period = 30 degrees
                   np.sin(np.radians(longitudes*360/10)) +   # Period = 10 degrees
                   np.sin(np.radians(longitudes*360/3.2)) +  # Period = 3.2 degrees
                   np.sin(np.radians(longitudes*360/0.7))    # Period = 0.7 degrees
                  )

    # Compute the frequency bins used by the discrete FFT in units of cycles per degree
    freq = np.fft.fftfreq(ew_data.size, d=long_res_deg)
    freq[freq == 0] = 0.000001 # Avoid divide by zero error later

    # Start with the FFT of the original EW data
    orig_fft = np.fft.fft(ew_data)
    orig_power = orig_fft * orig_fft.conjugate()

    # High pass filter. Remove all frequencies less than
    #   hpf_period cycles per complete ring.
    # For hpf_period=30 this removes any features with a width more than 12 deg.
    hp_fft = orig_fft.copy()
    hp_fft[np.where((freq >= -hpf_period/360) & (freq <= hpf_period/360))[0]] = 0
    hp_power = hp_fft * hp_fft.conjugate()

    # Low pass filter. Remove all frequencies great than
    #   lpf_period cycles per complete ring.
    # For lpf_period=90 this removes any features with a width less than 4 deg,
    # which includes the Prometheus 3.2 deg feature.
    lp_fft = hp_fft.copy()
    lp_fft[np.where((freq <= -lpf_period/360) | (freq >= lpf_period/360))[0]] = 0
    lp_power = lp_fft * lp_fft.conjugate()

    # Now take the FFT of the smoothed data and subtract it from the original FFT.
    # This should result in the subtraction of the whole periodicity from the data.
    # final_fft = hp_fft - box_fft
    # final_ew = np.fft.ifft(final_fft).real

    final_ew = np.fft.ifft(lp_fft).real

    if plot:
        orig_ifft = np.fft.ifft(orig_fft)  # Inverse FFT
        longitudes = np.arange(ew_data.size) * long_res_deg
        ew_plot = plt.subplot(321)
        ew_plot.plot(longitudes, ew_data, lw=2, color='red')
        ew_plot.plot(longitudes, orig_ifft.real, lw=1, color='blue')
        ew_plot.set_xlabel('Longitude (degrees)')
        ew_plot.set_ylabel('Normalized I/F')
        ew_plot.set_xlim(0, 360)
        ew_plot.set_title('Original EW vs EW->FFT->IFFT')

        ft_plot = plt.subplot(322)
        ft_plot.plot(1 / freq, orig_power, 'green')
        ft_plot.set_xlabel('Period (degrees)')
        ft_plot.set_ylabel('Power')
        ft_plot.set_xlim(0, 90)
        ft_plot.set_title('Original EW Power Spectrum')

        hp_filt = plt.subplot(323)
        hp_filt.plot(1 / freq, hp_power)
        hp_filt.set_xlabel('Period (degrees)')
        hp_filt.set_ylabel('Power')
        hp_filt.set_xlim(0, 90)
        hp_filt.set_title(
            f'Power Spectrum High-Pass Filtered ({360/hpf_period:.2f} deg/cycle)')

        hp_filt = plt.subplot(324)
        hp_filt.plot(1 / freq, lp_power)
        hp_filt.set_xlabel('Period (degrees)')
        hp_filt.set_ylabel('Power')
        hp_filt.set_xlim(0, 90)
        hp_filt.set_title(
            f'Power Spectrum HP & Low-Pass Filtered ({360/lpf_period:.2f} deg/cycle)')

        ew_plot = plt.subplot(3, 2, (5,6))
        ew_plot.plot(longitudes, np.fft.ifft(hp_fft), lw=1, color='blue', label='HPF')
        ew_plot.plot(longitudes, final_ew, lw=1, color='red', label='HPF&LPF')
        ew_plot.set_xlabel('Longitude (degrees)')
        ew_plot.set_ylabel('Normalized I/F')
        ew_plot.set_xlim(0, 360)
        ew_plot.set_title('HP Filtered EW with Box Filter')

        plt.tight_layout()
        plt.show()

    return final_ew

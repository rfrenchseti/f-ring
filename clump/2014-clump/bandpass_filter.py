'''
Author: Shannon Hicks
Bandpass filter for the EW profiles - this eliminates all of the mid-range frequencies
including Prometheus' 3.2 degree perturbance. 

Note that 3.2 is the PERIOD of the oscillation. The actual FREQUENCY is 112.5 (360/3.2)
The resulting profile is everything EXCEPT prometheus' interference.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from cwt import SDG, Morlet, cwt, icwt


def fft_filter(options, ew_data, plot = False):
        long_res = 360./len(ew_data)
        longitudes = np.arange(0,360, long_res) #18,000 element array
        
        ft = np.fft.fft(ew_data)
        power_ft = np.fft.fft(ew_data -np.mean(ew_data))
        
        ift = np.fft.ifft(ft)
        power = power_ft*power_ft.conjugate()
        
        freq = np.fft.fftfreq(int(longitudes.shape[0]), long_res)*360  # Weird int() cast is because of a problem with 64-bit Python on Windows
        
    #    ift_sub = np.fft.ifft(ft_sub)
        power = ft*ft.conjugate()
        
        #PLOT SET 1: Original data and IFT, FFT, POWER
        if plot:
            fit = np.max(power)/freq
            ew_plot = plt.subplot(311)
            ew_plot.plot(longitudes, ew_data, 'red')
            ew_plot.plot(longitudes, ift, 'blue')
     
            ft_plot = plt.subplot(312)
            plt.plot(freq, ft.real, 'green')
            
            p_plot = plt.subplot(313)
            p_plot.plot(freq, power - fit, 'red')
            p_plot.plot(freq, fit, 'blue')
        
            plt.show()
        
        #HIGH PASS FILTER EVERYTHING (remove lower frequencies)
        ft_sub = np.fft.fft(ew_data)
        lose = np.where((freq > -50) & (freq < 50))[0]
        ft_sub[lose] = 0
        
        #LOW PASS FILTER - USE A BOXCAR WINDOWING FUNCTION (removes higher frequencies)
        ift_sub = np.fft.ifft(ft_sub)
        #repeat the signal so that the windowing function doesn't create edge effects
        ift_sub = np.tile(ift_sub, 3)
        box = signal.boxcar(5)
        
        smooth = np.convolve(box/box.sum(), ift_sub, mode = 'valid')
        smooth = smooth[(len(longitudes)-((len(box)/2))):(-(len(longitudes)-(len(box)/2)))]

        freq_ratio = len(freq)/len(smooth)
        
        #fft of the smooth array should show the 3.2 signal and harmonics
        result = np.fft.fft(smooth)
        new_freq = np.fft.fftfreq(len(smooth), long_res)*360
        pow = result*result.conjugate()
        
        if plot:
            low_filt = plt.subplot(411)
            low_filt.plot(freq, ft_sub)
            
            smooth_ft = plt.subplot(412)
            smooth_ft.plot(smooth)
            
            smooth_ew = plt.subplot(413)
            smooth_ew.plot(longitudes,(ew_data - np.mean(ew_data)) - smooth, 'red')
            smooth_ew.plot(longitudes, smooth, 'blue')
            
            res_p = plt.subplot(414)
            res_p.plot(new_freq, result)
            plt.show()
        
        #now take the ft of the smoothed data and subtract it from the original ft
        #should result in the subtraction of the whole periodicity from the data
        
        final_ft = ft - result
        final = np.fft.ifft(final_ft)
        
        if plot:
            final_plot = plt.subplot(211)
            final_plot.plot(longitudes,final, 'blue')
            
            ew_fin = plt.subplot(212)
            ew_fin.plot(longitudes, ew_data - np.mean(ew_data), 'red')
            plt.show()
            
        return (final)


#NOTE: CWT_FILTER DOES NOT DO WHAT WE WANT IT TO DO (don't use it).
def cwt_filter(options, MosaicData, ew_data, plot = False):
    
    longitudes = np.arange(0,9000)*.04 #9,000 element array
    long_res = 0.04
    data = (ew_data - ew_data.mean())
#    data = np.tile(data, 3)
#    data = data[len(data):-(len(data)-1)]

    
#    for i in range(1,len(data)):
#        if data.mask[i]:
#            data[i] = data[i-1] 
            
    # create the scales at which you wish to do the high pass filter (let low frequencies through)
    min_scale = .1/2 # Degrees
    max_scale = 100./2 # Degrees
    scales = np.arange(min_scale/long_res, max_scale/long_res, .04/long_res)
    
    # initialize the mother wavelet
    mother_wavelet = SDG(len_signal = len(ew_data), #pad_to = np.power(2,10),
                         scales = scales, normalize = True)
    
    # perform continuous wavelet transform on `data` using `mother_wavelet`
    hp_wave = cwt(data, mother_wavelet)
    ew_data_bp = icwt(hp_wave) #result should be all of the long wavelengths.
    
    if plot:
        hp_plot = plt.subplot(412)
        hp_plot.plot(longitudes, ew_data_bp, 'blue')
          
        ew_original = plt.subplot(411)
        ew_original.plot(longitudes, (ew_data/np.max(ew_data)), 'red')
        
        
        plt.show()
        
    return (ew_data_bp)

#for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
#    (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
#    ew_data = np.load(ew_data_filename+'.npy')
#
#if options.fft_filter: 
#    fft_filter(options, mosaicdata, ew_data, plot = True)
#    
#if options.cwt_filter:
#    cwt_filter(options, MosaicData)
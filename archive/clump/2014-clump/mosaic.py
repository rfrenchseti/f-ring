import os
import sys
import numpy as np
import numpy.ma as ma
import vicar as vic
import ringimage as ri
import cspice
import ringutil

'''
Merge_obs is a series of functions used to create mosaics. 

ReadIMG: Reads in vicar files and returns a tuple containing the image, and various meta data.

MergeIntoMosaic: Reads in each image array and merges it into the mosaic. As each subsequent image is added,
the program checks to see which radial slice has better resolution. If it determines that the new slice is better,
it replaces the current data with the new.

MergeEWs: Background subtraction on the entire mosaic.

GetEWsfromObs: This is the function that is called to run the first three. It returns various statistical data in a list.
There are several other programs in this suite that call this function to either access or store data. 
''' 

global_debug = False

def LongitudeToIndex(longitude, longitude_resolution):
    return int(longitude/longitude_resolution)

def IndexToLongitude(index, longitude_resolution):
    return float(index) * longitude_resolution

def RadiusToIndex(radius, radius_start, radius_resolution):
    return int((radius-radius_start)/radius_resolution+0.5)

def IndexToRadius(radius_index, radius_resolution):
    return float(radius_index)*radius_resolution + 137500

def ReadIMG(image_filename):
    # Get info from image header 
    vicar_data = vic.VicarImage.FromFile(image_filename)
    img_array = vicar_data.Get2dArray()
    img_midtime = vicar_data['IMAGE_MID_TIME'][:-1]
    img_ET = cspice.utc2et(img_midtime)
    valid_longitudes = np.array([float(x) for x in vicar_data['LONGITUDES_SAVED']]) # In degrees!
    img_longitude_step = float(vicar_data['LONGITUDE_STEP'])        
    img_resolutions = np.array(vicar_data['RADIAL_RESOLUTION'])

    # Determine shift for image to be at 2007-1-1 position for mosaic
    longitude_shift = ringutil.ComputeLongitudeShift(img_ET)
    valid_longitudes = (valid_longitudes+longitude_shift) % 360.
    
    img_emission_angle = vicar_data['EMISSION_ANGLE']
    img_incidence_angle = vicar_data['INCIDENCE_ANGLE']
    img_phase_angle = vicar_data['PHASE_ANGLE']

    return (valid_longitudes, img_longitude_step, img_resolutions,
            img_array, img_ET, img_emission_angle, img_incidence_angle, img_phase_angle)

def MergeImageIntoMosaic(mosaic_img, img_array, valid_longitudes, longitude_resolution, img_resolutions,
                         img_image_number, img_ET, img_emission_angle, img_incidence_angle, img_phase_angle,  
                         mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers, mosaic_ETs, 
                         mosaic_emission_angles, mosaic_incidence_angles, mosaic_phase_angles,
                         mosaic_num_zeros_arr):
    
    # Merge an image into the mosaic
    # If we have more valid radii, definitely merge
    # If we have the same number of valid radii, choose the image with the higher resolution
    for c in range(valid_longitudes.size):
        column = img_array[:, c]
        output_column = LongitudeToIndex(valid_longitudes[c], longitude_resolution)
        num_zeros_in_col = len(np.where(column == 0)[0])
        if (mosaic_num_zeros_arr[output_column] >= num_zeros_in_col or
            (mosaic_num_zeros_arr[output_column] == num_zeros_in_col and
             mosaic_resolutions[output_column] > img_resolutions[c])):
            mosaic_img[:, output_column] = column
            # Recompute the longitude because the one from the reprojected image is not on longitude-resolution boundaries
            mosaic_longitudes[output_column] = IndexToLongitude(output_column, longitude_resolution) # valid_longitudes[c]
            mosaic_resolutions[output_column] = img_resolutions[c]
            mosaic_image_numbers[output_column] = img_image_number
            mosaic_ETs[output_column] = img_ET
            mosaic_emission_angles[output_column] = img_emission_angle
            mosaic_incidence_angles[output_column] = img_incidence_angle
            mosaic_phase_angles[output_column] = img_phase_angle
            mosaic_num_zeros_arr[output_column] = num_zeros_in_col

def MakeMosaic(file_list, radius_start, radius_end, radius_resolution, longitude_resolution):
    num_longitudes = LongitudeToIndex(360., longitude_resolution)
    num_radii = RadiusToIndex(radius_end+radius_resolution, radius_start, radius_resolution)
    mosaic_img = np.zeros([num_radii, num_longitudes])
    mosaic_longitudes = np.zeros(num_longitudes)-10 # Invalid = -10 
    mosaic_resolutions = np.zeros(num_longitudes)+1e37 # Extremely large resolution 
    mosaic_image_numbers = np.zeros(num_longitudes, dtype=np.int)-10  # Invalid = -10
    mosaic_ETs = np.zeros(num_longitudes)-10  # Invalid = -10
    mosaic_emission_angles = np.zeros(num_longitudes)
    mosaic_incidence_angles = np.zeros(num_longitudes)
    mosaic_phase_angles = np.zeros(num_longitudes)
    num_zeros_arr = np.zeros(num_longitudes)+1e37 # Extremely large number of zeros
    
    for img_num, file_name in enumerate(file_list):
        print 'Merging', file_name
        (valid_longitudes, img_longitude_step, img_resolutions,
         img_array, img_ET, img_emission_angle, img_incidence_angle,
         img_phase_angle) = ReadIMG(file_name)
        
        assert img_longitude_step == longitude_resolution
         
        MergeImageIntoMosaic(mosaic_img, img_array, valid_longitudes, longitude_resolution, img_resolutions,
                             img_num, img_ET, img_emission_angle, img_incidence_angle, img_phase_angle,  
                             mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers, mosaic_ETs, 
                             mosaic_emission_angles, mosaic_incidence_angles, mosaic_phase_angles,
                             num_zeros_arr)

    return (mosaic_img, mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers, mosaic_ETs, 
            mosaic_emission_angles, mosaic_incidence_angles, mosaic_phase_angles)

def ComputeEWs(mosaic_img, radius_resolution, longitude_resolution):
    num_longitudes = mosaic_img.shape[1]
    
    ew_arr = np.zeros(num_longitudes)-10
    center_of_brightness_arr = np.zeros(num_longitudes)-10

    row_weights_for_ctr = np.arange(stretchimg_array.shape[0])
    for output_column in range(num_longitudes):
        column = mosaic_img[:, output_column]
        ew_raw = np.sum(column)
        ew = ew_raw * radius_resolution
        center_of_brightness = np.sum(column*row_weights_for_ctr) / ew_raw
        ew_arr[output_column] = ew
        center_of_brightness_arr[output_column] = center_of_brightness
                   
    mosaic = None

#COMPUTE STATISTICS
    print '**********************************'*10
    print 'Image Dictionary', img_dict
    total_images = num_files
#    print total_images
    
    #get first and last image numbers
    
    first_image_name = first_image_name[:first_image_name.find('_')]
    last_image_name = last_image_name[:last_image_name.find('_')]
    first_last_img = (first_image_name, last_image_name)
    
    #get date  - if the first element is empty, iterate through until there's a date
    d = 0
    if dates[d] != -10:
        cal_date = (cspice.et2utc(dates[d], 'C', 0))[0:-9]
    else:
        while dates[d] == -10:
            d += 1
            cal_date = (cspice.et2utc(dates[d], 'C', 0))[0:-9]

        
#    print cal_date
    
    #In order to compute statistics, the data needs to be in one long lists, instead of separate lists. 
    #These names are going to change several times, make sure to track them carefully
    #I'm sure there's a better way to do this - if you think of one, please change it!
    
    long_list = []
    ew_list = []
    ctr_bright_list = []
    res_list = []
    img_num_list = []
    dates_list = []
    em_angles_list = []
    phase_angles_list = []
    inc_angles_list = []
      
    
    temp_longs = np.zeros(18002) -10
    temp_longs[1:18001] = longitudes
    
    invalids = np.where(temp_longs == -10)[0]
    z_index = invalids[1:] - invalids[0:-1]
    not_zeroes = np.where(z_index >1)[0]
    
#    print 'check zeroes:', invalids, z_index, not_zeroes
#    print 'temp_longs', temp_longs
    
    for idx in not_zeroes:
        long_list.append(temp_longs[invalids[idx]+1: invalids[idx +1]])
        ew_list.append(ew_arr[invalids[idx]:invalids[idx+1]-1])
        ctr_bright_list.append(center_of_brightness_arr[invalids[idx]:invalids[idx+1]-1])
        res_list.append(resolutions[invalids[idx]:invalids[idx+1]-1])
        dates_list.append(dates[invalids[idx]:invalids[idx+1]-1])
        img_num_list.append(image_num_arr[invalids[idx]:invalids[idx+1]-1])
        em_angles_list.append(emission_angles[invalids[idx]:invalids[idx+1]-1])
        phase_angles_list.append(phase_angles[invalids[idx]:invalids[idx+1]-1])
        inc_angles_list.append(incidence_angles[invalids[idx]:invalids[idx+1]-1]) 
#    print 'LONGITUDES:', longitudes
#    print 'LONG_LIST:', long_list
    #append EW list to compute statistics :) 
    all_ew = []
    all_res = []
    all_cob = []
    all_phase_angles = []
    all_emission_angles = []
    all_incidence_angles = []
    
    for piece in ew_list:
        all_ew.extend(piece)
    
    for res in res_list:
        all_res.extend(res)
        
    for cob in ctr_bright_list:
        all_cob.extend(cob)
    
    for phase in phase_angles_list:
        all_phase_angles.extend(phase)
        
    for em in em_angles_list:
        all_emission_angles.extend(em)
    
    for ia in inc_angles_list:
        all_incidence_angles.extend(ia)
    
    
    min_ew = np.min(all_ew)
    max_ew = np.max(all_ew)
    stddev_ew = np.std(all_ew)
    mean_ew = np.mean(all_ew)
    
    min_res = np.min(all_res)
    max_res = np.max(all_res)
    stddev_res = np.std(all_res)
    mean_res = np.mean(all_res)
    
    min_cob = np.min(all_cob)
    max_cob = np.max(all_cob)
    stddev_cob = np.std(all_cob)
    mean_cob = np.mean(all_cob)
    
    min_ea = np.min(all_emission_angles)
    max_ea = np.max(all_emission_angles)
    mean_ea = np.mean(all_emission_angles)
    stddev_ea = np.std(all_emission_angles)
    
    min_pa = np.min(all_phase_angles)
    max_pa = np.max(all_phase_angles)
    mean_pa = np.mean(all_phase_angles)
    stddev_pa = np.std(all_phase_angles)
    
    min_ia = np.min(all_incidence_angles)
    max_ia = np.max(all_incidence_angles)
    mean_ia = np.mean(all_incidence_angles)
    stddev_ia = np.std(all_incidence_angles)
    
    ew_stats = [min_ew, max_ew, mean_ew, stddev_ew]
#    ew_stats = []
    res_stats = [min_res, max_res, mean_res, stddev_res]
#    res_stats = []
    cob_stats = [min_cob, max_cob, mean_cob, stddev_cob]
    
    ea_stats = [min_ea, max_ea, mean_ea, stddev_ea]
    pa_stats = [min_pa, max_pa, mean_pa, stddev_pa]
    ia_stats = [min_ia, max_ia, mean_ia, stddev_ia]
    #pack longitudes and EWs into a list of tuples
    data_list = []
    cob_list = []
    long_ranges = []
    image_num_list = []
    date_list = []
    em_ang_list = []
    phase_ang_list = []
    inc_ang_list = []
    
    #create the data array of tuples, but also run min and max functions on each range
    #NOTE: RUNNING THE BACKGROUND SUBTRACTION CAUSES GAPS IN THE LONGITUDES WHEN DATA IS MASKED
        #WILL FIX LATER AFTER THE BACKGROUND IS OPTIMIZED. 
    for i in range(len(long_list)):
        data_list.append([long_list[i], ew_list[i]])
        cob_list.append([long_list[i], ctr_bright_list[i]])
        long_ranges.append((int(np.min(long_list[i])), int(np.max(long_list[i]))))
        image_num_list.append([long_list[i], img_num_list[i]])
        date_list.append([long_list[i], dates_list[i]])
        em_ang_list.append([long_list[i], em_angles_list[i]])
        phase_ang_list.append([long_list[i], phase_angles_list[i]])
        inc_ang_list.append([long_list[i], inc_angles_list[i]])
#    j = 0
#    for k in range(len(long_ranges)-1):
#        dif = long_ranges[j+1][0] - long_ranges[j][1]
#        if dif <= 5:
#            long_ranges[j] = (long_ranges[j][0],long_ranges[j+1][1])
#            del long_ranges[j+1]
#        else:
#            j += 1
##Long_ranges is now a list of tuples, where any breaks larger than 5 degrees are accounted for. Now to make it pretty for the table....
#    ranges = []
#    m = 0
#    for set in long_ranges:
#        expand = str(long_ranges[m][0]) + '-' + str(long_ranges[m][1])
#        ranges.append(expand)
#        m +=1    
#IF LONGITUDE_WRAP = TRUE
    
    if longitude_wrap:
        if longitudes[0] != -10 and longitudes[-1] != -10:
            #combine the contiguous data
            if len(data_list) > 1:
                
                data_list[0][0] = np.append(data_list[-1][0], data_list[0][0])
                data_list[0][1] = np.append(data_list[-1][1], data_list[0][1])
                del data_list[-1]
                
    return [data_list, cob_list, image_num_list, date_list, long_ranges, total_images, ew_stats,
             res_stats, cob_stats, first_last_img, cal_date, img_dict, 
             em_ang_list, ea_stats, phase_ang_list, pa_stats, inc_ang_list, ia_stats]

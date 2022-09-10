#!/bin/bash

#slice size (integer)
size=0 #if 0,1 bin for entire mosaic)

#running all of the different core widths
#updated to run to edge of image

#core_rins=(140170 140120 140070 140020 139970 139920 139870 139820 139770 139720 139670 139620 139570 139520 139470 139420 139370 139320 139270 139220)
#core_routs=(140270 140320 140370 140420 140470 140520 140570 140620 140670 140720 140770 140820 140870 140920 140970 141020 141070 141120 141170 141220)

#define function for running create_ew.py
#UPDATED version, running create ew / dump csv in the same python script
run_create_ew () {

	# $1 = region
	# $2 = rin
	# $3 = rout

	#create region/radius directories if they do not exist
	mkdir -p ~/REU_2022/data/dump_ew_csv/$1/rin$2_rout$3

        #reset the environment variable for EW_DIR based on region, radii being used
        #unset EW_DIR
        #ew_dir_path=~/REU_2022/data/dump_ew_csv/$1/rin$2_rout$3
        #export EW_DIR=$ew_dir_path

	#set path to export csv file to
	ew_csv_path=~/REU_2022/data/dump_ew_csv/$1/rin$2_rout$3/slice${size}_ew_stats.csv

        #run create_ew.py, calculating ew values and dumping stats into csv at same time
	python ~/REU_2022/src/create_ews.py --ew-inner-radius $2 --ew-outer-radius $3 --slice-size $size --output-csv-filename $ew_csv_path
}



#core
region=core
#updated version, using loop to calculate core values
for ((r1=50;r1<1001;r1+=50)); do
        #assign rin and rout values
        let rin=140220-$r1
        let rout=140220+$r1

	#run ew function
	run_create_ew $region $rin $rout
	echo $region R_in=$rin R_out=$rout completed
done

#run inner/outer regions (just the ones you need for the 3-region one, for the moment)

#region=inner
#run_create_ew $region 139220 140120
#run_create_ew $region 139220 140119

#region=outer
#run_create_ew $region 140320 141220
#run_create_ew $region 140320 141220

#region=core
#run_create_ew $region 140120 140319

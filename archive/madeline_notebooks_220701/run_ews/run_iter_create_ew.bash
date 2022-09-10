#!/bin/bash


#running the 10-km iterative create_ew (ultimately to get tau as a function of radius) 
#image max extent on either side runs 139,220 - 141,220 km (+/- 1000 km from center 140220)

#slice size (integer)
size=0 #if 0,1 bin for entire mosaic)
region=iter_10

#define function for running create_ew.py
#UPDATED version, running create ew / dump csv in the same python script
run_create_ew () {

        # $1 = region
        # $2 = rin
        # $3 = rout

        #create region directory if it does not exist
        mkdir -p ~/REU_2022/data/dump_ew_csv/$1

        #create radius directory if it does not exist
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

#run the iterative region runs
for ((r1=139220;r1<141220;r1+=10)); do
	#assign rin and rout values
	rin=$r1
	let rout=$r1+9
	
	run_create_ew $region $rin $rout
	echo $region R_in=$rin R_out=$rout completed
done










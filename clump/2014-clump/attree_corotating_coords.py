import os
import sys
import ringutil
import cspice

new_filename = 'attree_minijets_corotating.txt'
new_fp = os.path.join(ringutil.ROOT, 'Attree_analysis', new_filename)
new_file = open(new_fp, 'w')

old_filename = 'attree_minijet_catalogue.csv'
old_fp = os.path.join(ringutil.ROOT, 'Attree_analysis', old_filename)
old_file = open(old_fp, 'r')

filelines = old_file.readline()
filelines = filelines.rsplit('\r')

for a, line in enumerate(filelines):
    
    if a == 0:
        print a
        print line
        new_file.write(line + '\n')
    if a != 0:
        print a
        print line
#        print line.rsplit(',')
        designation, date, exposure, phase, tip_rad, tip_long, base_rad ,base_long, jet_class = line.rsplit(',')
        
        tip_long = float(tip_long)
        base_long = float(base_long)
        date = list(date)
        date[8] = '/'
        date = ''.join(date)
#        print date
        et = cspice.utc2et(date)
        tip_long = '%.3f'%(ringutil.InertialToCorotating(tip_long, et))
        base_long = '%.3f'%( ringutil.InertialToCorotating(base_long, et))
        
        new_line = designation + ',' + date + ',' + exposure + ',' + phase +',' + tip_rad +',' + tip_long + ',' + base_rad + ',' + base_long + ',' + jet_class + '\n'
        print new_line
        
        new_file.write(new_line)
        

new_file.close()
old_file.close()
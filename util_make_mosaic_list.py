import argparse
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.ndimage as nd

import f_ring_util.f_ring as f_ring


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)


for obs_id in f_ring.enumerate_obsids(arguments):
    print(obs_id)
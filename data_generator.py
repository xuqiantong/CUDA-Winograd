from __future__ import print_function
import sys
import numpy as np
import re
import time
import difflib
import array
import requests
import os
import shutil
import scipy.spatial.distance as spd
import numpy as np
from numpy.random import *
from random import randint
import random
import matplotlib.pyplot as plt
from scipy import misc


def bias_generator(output_channel = 128):
	bias = (np.array(rand(output_channel))-0.5).astype(np.float32)
	des = open("data/bias_" + str(output_channel) + ".bin", "wb")
	des.write(bias)

	bnScale = (np.array(rand(output_channel))-0.5).astype(np.float32)
	des = open("data/bnScale_" + str(output_channel) + ".bin", "wb")
	des.write(bnScale)

	bnBias = (np.array(rand(output_channel))-0.5).astype(np.float32)
	des = open("data/bnBias_" + str(output_channel) + ".bin", "wb")
	des.write(bnBias)

	eMean = (np.array(rand(output_channel))-0.5).astype(np.float32)
	des = open("data/eMean_" + str(output_channel) + ".bin", "wb")
	des.write(eMean)

	eVar = (np.array(rand(output_channel))*3 + 5).astype(np.float32)
	des = open("data/eVar_" + str(output_channel) + ".bin", "wb")
	des.write(eVar)

	eps = 1e-5
	bnScale_winograd = bnScale / np.sqrt(eVar + eps)
	des = open("data/bnScale_winograd_" + str(output_channel) + ".bin", "wb")
	des.write(bnScale_winograd)
	bnBias_winograd = bnBias - bnScale*eMean / np.sqrt(eVar + eps)
	des = open("data/bnBias_winograd_" + str(output_channel) + ".bin", "wb")
	des.write(bnBias_winograd)

def input_generator(input_channel = 128, feature_map_size = 14, padding = 1):
	parameters = (feature_map_size + 2*padding)*(feature_map_size + 2*padding) * input_channel
	a = (np.array(rand(parameters))-0.5).astype(np.float32)
	des = open("data/input_" + str(feature_map_size) + '_' + str(padding) + '_' + str(input_channel) + ".bin", "wb")
	des.write(a)

def weight_generator(input_channel = 128, output_channel = 128):
	### Weights_NCHW
	parameters = input_channel*output_channel * 3*3
	in_ = (np.array(rand(parameters))-0.5).astype(np.float32)

	des = open("data/weight_NCHW_" + str(input_channel) + '_' + str(output_channel) + ".bin", "wb")
	des.write(in_)

	### Weights_Winograd
	in_ = in_.reshape(input_channel*output_channel, 3,3)
	G = np.array([[0.25,0,0], [-1.0/6,-1.0/6,-1.0/6], [-1.0/6,1.0/6,-1.0/6], [1.0/24,1.0/12,1.0/6], [1.0/24,-1.0/12,1.0/6], [0,0,1]])

	out_ = [0] * input_channel*output_channel * 6*6
	for i in range(output_channel):
		for j in range(input_channel):
			b = np.dot(G, in_[i*input_channel+j])
			b = np.dot(b, G.transpose())
			offset = j*output_channel+i
			for x in range(6):
				for y in range(6):
					out_[((x*6+y) * input_channel*output_channel) + offset] = b[x][y]

	des = open("data/weight_winograd_" + str(input_channel) + '_' + str(output_channel) + ".bin", "wb")
	des.write(np.array(out_).astype(np.float32))

def onebyone_generator(input_channel = 256, output_channel = 1024, feature_map_size = 14):
	parameters = feature_map_size*feature_map_size * output_channel
	a = ((np.array(rand(parameters))-0.5)*40).astype(np.float32)
	des = open("data/input_one_" + str(feature_map_size) + '_' + str(output_channel) + ".bin", "wb")
	des.write(a)

	parameters = input_channel * output_channel
	a = ((np.array(rand(parameters))-0.5)*40).astype(np.float32)
	des = open("data/weight_one_" + str(output_channel) + ".bin", "wb")
	des.write(a)

	bnScale = ((np.array(rand(output_channel))-0.5)*40).astype(np.float32)
	des = open("data/bnScale_one_" + str(output_channel) + ".bin", "wb")
	des.write(bnScale)

	bnBias = ((np.array(rand(output_channel))-0.5)*40).astype(np.float32)
	des = open("data/bnBias_one_" + str(output_channel) + ".bin", "wb")
	des.write(bnBias)

	eMean = ((np.array(rand(output_channel))-0.5)*40).astype(np.float32)
	des = open("data/eMean_one_" + str(output_channel) + ".bin", "wb")
	des.write(eMean)

	eVar = (np.array(rand(output_channel))*20 + 5).astype(np.float32)
	des = open("data/eVar_one_" + str(output_channel) + ".bin", "wb")
	des.write(eVar)

	eps = 1e-5
	bnScale_winograd = bnScale / np.sqrt(eVar + eps)
	des = open("data/bnScale_myKernel_one_" + str(output_channel) + ".bin", "wb")
	des.write(bnScale_winograd)
	bnBias_winograd = bnBias - bnScale*eMean / np.sqrt(eVar + eps)
	des = open("data/bnBias_myKernel_one_" + str(output_channel) + ".bin", "wb")
	des.write(bnBias_winograd)


if __name__ == '__main__':
	#bias_generator(output_channel = 256)
	print('Biases generated')

	#input_generator(input_channel = 256)
	print('Input generated')

	#weight_generator(256, 256)
	print('Weights generated')

	onebyone_generator()
	print('Parameters for 1*1 conv generated')


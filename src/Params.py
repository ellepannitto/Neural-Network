import math
import random

def sigmoid_function (x):
		return 1.0 / (1 + math.exp(x*SLOPE) )

def derivative_sigmoid_function (x):
		return -SLOPE*math.exp(SLOPE*x) / ((1 + math.exp(x*SLOPE) )**2)

ActivationFunctions = {
	"sigmoid": lambda x: sigmoid_function(x),
	"sigmoid_der": lambda x: derivative_sigmoid_function(x),
	"id": lambda x: x,
	"id_der": lambda x: 1
}

def unit_weigth_initializer ():
	return 1

def constant_weigth_initializer_initializer ():
	
	def constant_weigth_initializer ():
		constant_weigth_initializer.nxt += 1
		return constant_weigth_initializer.weight_to_assign[constant_weigth_initializer.nxt]
		
	constant_weigth_initializer.weight_to_assign = [0.15, 0.2, 0.35, 0.25, 0.30, 0.35, 0.40, 0.45, 0.6, 0.50, 0.55, 0.6]
	constant_weigth_initializer.nxt = -1
	return constant_weigth_initializer 

WEIGHTS = [-700, 700]
	
def random_weigth_initializer ():
	return random.uniform ( WEIGHTS[0], WEIGHTS[1] ) * 1.0 / 1000 


SLOPE = -1
ETA = 0.04

MAX_EPOCH = 2000

ALPHA = 0

LAMBDA = 0.0001

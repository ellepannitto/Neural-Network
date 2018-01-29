import math
import random

random.seed(0)

def sigmoid_function (x):
	try:
		return 1.0 / (1 + math.exp(x*SLOPE) )
	except Exception as e: 
		print ("error in sigmoid function x:{}".format(x))
		print (e)
		
	return 0

def derivative_sigmoid_function (x):
	try:
		return -SLOPE*math.exp(SLOPE*x) / ((1 + math.exp(x*SLOPE) )**2)
	except Exception as e:
		print ("error in derivative sigmoid function x:{}".format(x))
		print (e)
	return 0
		
ActivationFunctions = {
	"sigmoid": sigmoid_function,
	"sigmoid_der": derivative_sigmoid_function,
	"id": lambda x: x,
	"id_der": lambda x: 1
}

def unit_weigth_initializer ():
	return 1

def constant_weigth_initializer_initializer ():
	
	def constant_weigth_initializer ():
		constant_weigth_initializer.nxt += 1
		return constant_weigth_initializer.weight_to_assign[constant_weigth_initializer.nxt]
		
	constant_weigth_initializer.weight_to_assign = [0.15, 0.2, 0.35, 0.25, 0.30, 0.35, 0.4, 0.45, 0.6, 0.5, 0.55, 0.6]
	constant_weigth_initializer.nxt = -1
	return constant_weigth_initializer 

WEIGHTS = [-700, 700]
	
def random_weigth_initializer ():
	return random.uniform ( WEIGHTS[0], WEIGHTS[1] ) * 1.0 / 1000


SLOPE = -1

ETA = 1
ALPHA = 0.8
LAMBDA = 0.002

ETA_DECAY=True
ETA_RANGE=[0.5, 1]
ETA_DECREASING_PERIOD=3/4

NUM_TRIALS_PER_CONFIGURATION = 3
MAX_EPOCH = 200

MINIBATCH=True
#set MINIBATCH_SAMPLE=1 for online version
MINIBATCH_SAMPLE=35

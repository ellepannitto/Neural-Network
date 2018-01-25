
class MSELoss:
	
	def __init__(self):
		self.loss = 0.0
		
	def update (self, predicted, gold):
		for i in range(len(predicted)):
			self.loss += 0.5*((predicted[i] - gold[i])**2)
			
		#~ print	

class Accuracy:
	
	def __init__ (self):
		self.correctly_predicted = 0
		self.num_patterns = 0
	
	def update ( self, predicted, gold ):
		#~ print ("predicted {}".format(predicted))
		#~ print ("gold {}".format(gold))
		self.num_patterns += 1
		self.correctly_predicted += 1 if all ( [ (o-0.5)*(y-0.5)>0 for o,y in zip (predicted, gold) ] ) else 0
	
	def get ( self):
		return self.correctly_predicted*1.0 / self.num_patterns

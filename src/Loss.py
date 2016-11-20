class Loss:
	
	def __init__(self):
		self.loss = 0.0
		
	def update (self, predicted, gold):
		for i in range(len(predicted)):
			self.loss += 0.5*((predicted[i] - gold[i])**2)
			
		#~ print	

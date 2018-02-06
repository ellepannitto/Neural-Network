'''
  Loads the three Monk’s problem datasets (binary classification tasks),
  one-hot encoding the categorical features.
  
  see also: Sebastian B Thrun et al. The monk’s problems: A performance comparison of different learning algorithms. Tech. rep. 1991.
  
'''

import OneHotEncoder

monk1_train = [[1,1,1,1,1,3,1], [1,1,1,1,1,3,2], [1,1,1,1,3,2,1], [1,1,1,1,3,3,2], [1,1,1,2,1,2,1], [1,1,1,2,1,2,2], [1,1,1,2,2,3,1], [1,1,1,2,2,4,1], [1,1,1,2,3,1,2], [1,1,2,1,1,1,2], [0,1,2,1,1,2,1], [0,1,2,1,1,3,1], [0,1,2,1,1,4,2], [1,1,2,1,2,1,1], [0,1,2,1,2,3,1], [0,1,2,1,2,3,2], [0,1,2,1,2,4,2], [0,1,2,1,3,2,1], [0,1,2,1,3,4,2], [0,1,2,2,1,2,2], [0,1,2,2,2,3,2], [0,1,2,2,2,4,1], [0,1,2,2,2,4,2], [0,1,2,2,3,2,2], [0,1,2,2,3,3,1], [0,1,2,2,3,3,2], [0,1,3,1,1,2,1], [0,1,3,1,1,4,1], [0,1,3,1,2,2,1], [0,1,3,1,2,4,1], [1,1,3,1,3,1,2], [0,1,3,1,3,2,2], [0,1,3,1,3,3,1], [0,1,3,1,3,4,1], [0,1,3,1,3,4,2], [0,1,3,2,1,2,2], [1,1,3,2,2,1,2], [0,1,3,2,2,2,2], [0,1,3,2,2,3,2], [0,1,3,2,2,4,1], [0,1,3,2,2,4,2], [1,1,3,2,3,1,1], [0,1,3,2,3,2,1], [0,1,3,2,3,4,1], [0,1,3,2,3,4,2], [0,2,1,1,1,3,1], [0,2,1,1,1,3,2], [1,2,1,1,2,1,1], [1,2,1,1,2,1,2], [0,2,1,1,2,2,2], [0,2,1,1,2,3,1], [0,2,1,1,2,4,1], [0,2,1,1,2,4,2], [0,2,1,1,3,4,1], [0,2,1,2,1,2,2], [0,2,1,2,1,3,1], [0,2,1,2,1,4,2], [0,2,1,2,2,3,1], [0,2,1,2,2,4,2], [0,2,1,2,3,2,2], [0,2,1,2,3,4,1], [1,2,2,1,1,2,1], [1,2,2,1,1,2,2], [1,2,2,1,1,3,1], [1,2,2,1,2,3,2], [1,2,2,1,3,1,1], [1,2,2,1,3,1,2], [1,2,2,1,3,2,2], [1,2,2,1,3,3,2], [1,2,2,1,3,4,2], [1,2,2,2,1,1,1], [1,2,2,2,1,3,2], [1,2,2,2,1,4,1], [1,2,2,2,1,4,2], [1,2,2,2,2,2,1], [1,2,2,2,3,4,1], [1,2,3,1,1,1,1], [1,2,3,1,2,1,1], [0,2,3,1,2,3,1], [1,2,3,1,3,1,2], [0,2,3,1,3,3,1], [0,2,3,1,3,4,2], [0,2,3,2,1,3,2], [1,2,3,2,2,1,1], [1,2,3,2,2,1,2], [0,2,3,2,2,2,1], [0,2,3,2,3,3,2], [1,3,1,1,1,1,1], [1,3,1,1,1,1,2], [1,3,1,1,2,1,1], [0,3,1,1,2,2,2], [0,3,1,1,3,2,2], [1,3,1,2,1,1,1], [0,3,1,2,1,2,2], [0,3,1,2,2,2,2], [0,3,1,2,2,3,2], [0,3,1,2,3,2,2], [1,3,2,1,1,1,1], [0,3,2,1,1,4,2], [1,3,2,1,2,1,2], [0,3,2,1,2,4,2], [1,3,2,2,1,1,1], [1,3,2,2,1,1,2], [0,3,2,2,1,3,2], [1,3,2,2,3,1,1], [0,3,2,2,3,2,1], [0,3,2,2,3,4,1], [1,3,3,1,1,1,1], [1,3,3,1,1,2,1], [1,3,3,1,1,4,2], [1,3,3,1,2,3,2], [1,3,3,1,2,4,2], [1,3,3,1,3,1,2], [1,3,3,1,3,2,1], [1,3,3,1,3,2,2], [1,3,3,1,3,4,2], [1,3,3,2,1,1,1], [1,3,3,2,1,3,2], [1,3,3,2,1,4,1], [1,3,3,2,1,4,2], [1,3,3,2,3,1,2], [1,3,3,2,3,2,2], [1,3,3,2,3,3,2], [1,3,3,2,3,4,2]]
monk1_test = [[1,1,1,1,1,1,1], [1,1,1,1,1,1,2], [1,1,1,1,1,2,1], [1,1,1,1,1,2,2], [1,1,1,1,1,3,1], [1,1,1,1,1,3,2], [1,1,1,1,1,4,1], [1,1,1,1,1,4,2], [1,1,1,1,2,1,1], [1,1,1,1,2,1,2], [1,1,1,1,2,2,1], [1,1,1,1,2,2,2], [1,1,1,1,2,3,1], [1,1,1,1,2,3,2], [1,1,1,1,2,4,1], [1,1,1,1,2,4,2], [1,1,1,1,3,1,1], [1,1,1,1,3,1,2], [1,1,1,1,3,2,1], [1,1,1,1,3,2,2], [1,1,1,1,3,3,1], [1,1,1,1,3,3,2], [1,1,1,1,3,4,1], [1,1,1,1,3,4,2], [1,1,1,2,1,1,1], [1,1,1,2,1,1,2], [1,1,1,2,1,2,1], [1,1,1,2,1,2,2], [1,1,1,2,1,3,1], [1,1,1,2,1,3,2], [1,1,1,2,1,4,1], [1,1,1,2,1,4,2], [1,1,1,2,2,1,1], [1,1,1,2,2,1,2], [1,1,1,2,2,2,1], [1,1,1,2,2,2,2], [1,1,1,2,2,3,1], [1,1,1,2,2,3,2], [1,1,1,2,2,4,1], [1,1,1,2,2,4,2], [1,1,1,2,3,1,1], [1,1,1,2,3,1,2], [1,1,1,2,3,2,1], [1,1,1,2,3,2,2], [1,1,1,2,3,3,1], [1,1,1,2,3,3,2], [1,1,1,2,3,4,1], [1,1,1,2,3,4,2], [1,1,2,1,1,1,1], [1,1,2,1,1,1,2], [0,1,2,1,1,2,1], [0,1,2,1,1,2,2], [0,1,2,1,1,3,1], [0,1,2,1,1,3,2], [0,1,2,1,1,4,1], [0,1,2,1,1,4,2], [1,1,2,1,2,1,1], [1,1,2,1,2,1,2], [0,1,2,1,2,2,1], [0,1,2,1,2,2,2], [0,1,2,1,2,3,1], [0,1,2,1,2,3,2], [0,1,2,1,2,4,1], [0,1,2,1,2,4,2], [1,1,2,1,3,1,1], [1,1,2,1,3,1,2], [0,1,2,1,3,2,1], [0,1,2,1,3,2,2], [0,1,2,1,3,3,1], [0,1,2,1,3,3,2], [0,1,2,1,3,4,1], [0,1,2,1,3,4,2], [1,1,2,2,1,1,1], [1,1,2,2,1,1,2], [0,1,2,2,1,2,1], [0,1,2,2,1,2,2], [0,1,2,2,1,3,1], [0,1,2,2,1,3,2], [0,1,2,2,1,4,1], [0,1,2,2,1,4,2], [1,1,2,2,2,1,1], [1,1,2,2,2,1,2], [0,1,2,2,2,2,1], [0,1,2,2,2,2,2], [0,1,2,2,2,3,1], [0,1,2,2,2,3,2], [0,1,2,2,2,4,1], [0,1,2,2,2,4,2], [1,1,2,2,3,1,1], [1,1,2,2,3,1,2], [0,1,2,2,3,2,1], [0,1,2,2,3,2,2], [0,1,2,2,3,3,1], [0,1,2,2,3,3,2], [0,1,2,2,3,4,1], [0,1,2,2,3,4,2], [1,1,3,1,1,1,1], [1,1,3,1,1,1,2], [0,1,3,1,1,2,1], [0,1,3,1,1,2,2], [0,1,3,1,1,3,1], [0,1,3,1,1,3,2], [0,1,3,1,1,4,1], [0,1,3,1,1,4,2], [1,1,3,1,2,1,1], [1,1,3,1,2,1,2], [0,1,3,1,2,2,1], [0,1,3,1,2,2,2], [0,1,3,1,2,3,1], [0,1,3,1,2,3,2], [0,1,3,1,2,4,1], [0,1,3,1,2,4,2], [1,1,3,1,3,1,1], [1,1,3,1,3,1,2], [0,1,3,1,3,2,1], [0,1,3,1,3,2,2], [0,1,3,1,3,3,1], [0,1,3,1,3,3,2], [0,1,3,1,3,4,1], [0,1,3,1,3,4,2], [1,1,3,2,1,1,1], [1,1,3,2,1,1,2], [0,1,3,2,1,2,1], [0,1,3,2,1,2,2], [0,1,3,2,1,3,1], [0,1,3,2,1,3,2], [0,1,3,2,1,4,1], [0,1,3,2,1,4,2], [1,1,3,2,2,1,1], [1,1,3,2,2,1,2], [0,1,3,2,2,2,1], [0,1,3,2,2,2,2], [0,1,3,2,2,3,1], [0,1,3,2,2,3,2], [0,1,3,2,2,4,1], [0,1,3,2,2,4,2], [1,1,3,2,3,1,1], [1,1,3,2,3,1,2], [0,1,3,2,3,2,1], [0,1,3,2,3,2,2], [0,1,3,2,3,3,1], [0,1,3,2,3,3,2], [0,1,3,2,3,4,1], [0,1,3,2,3,4,2], [1,2,1,1,1,1,1], [1,2,1,1,1,1,2], [0,2,1,1,1,2,1], [0,2,1,1,1,2,2], [0,2,1,1,1,3,1], [0,2,1,1,1,3,2], [0,2,1,1,1,4,1], [0,2,1,1,1,4,2], [1,2,1,1,2,1,1], [1,2,1,1,2,1,2], [0,2,1,1,2,2,1], [0,2,1,1,2,2,2], [0,2,1,1,2,3,1], [0,2,1,1,2,3,2], [0,2,1,1,2,4,1], [0,2,1,1,2,4,2], [1,2,1,1,3,1,1], [1,2,1,1,3,1,2], [0,2,1,1,3,2,1], [0,2,1,1,3,2,2], [0,2,1,1,3,3,1], [0,2,1,1,3,3,2], [0,2,1,1,3,4,1], [0,2,1,1,3,4,2], [1,2,1,2,1,1,1], [1,2,1,2,1,1,2], [0,2,1,2,1,2,1], [0,2,1,2,1,2,2], [0,2,1,2,1,3,1], [0,2,1,2,1,3,2], [0,2,1,2,1,4,1], [0,2,1,2,1,4,2], [1,2,1,2,2,1,1], [1,2,1,2,2,1,2], [0,2,1,2,2,2,1], [0,2,1,2,2,2,2], [0,2,1,2,2,3,1], [0,2,1,2,2,3,2], [0,2,1,2,2,4,1], [0,2,1,2,2,4,2], [1,2,1,2,3,1,1], [1,2,1,2,3,1,2], [0,2,1,2,3,2,1], [0,2,1,2,3,2,2], [0,2,1,2,3,3,1], [0,2,1,2,3,3,2], [0,2,1,2,3,4,1], [0,2,1,2,3,4,2], [1,2,2,1,1,1,1], [1,2,2,1,1,1,2], [1,2,2,1,1,2,1], [1,2,2,1,1,2,2], [1,2,2,1,1,3,1], [1,2,2,1,1,3,2], [1,2,2,1,1,4,1], [1,2,2,1,1,4,2], [1,2,2,1,2,1,1], [1,2,2,1,2,1,2], [1,2,2,1,2,2,1], [1,2,2,1,2,2,2], [1,2,2,1,2,3,1], [1,2,2,1,2,3,2], [1,2,2,1,2,4,1], [1,2,2,1,2,4,2], [1,2,2,1,3,1,1], [1,2,2,1,3,1,2], [1,2,2,1,3,2,1], [1,2,2,1,3,2,2], [1,2,2,1,3,3,1], [1,2,2,1,3,3,2], [1,2,2,1,3,4,1], [1,2,2,1,3,4,2], [1,2,2,2,1,1,1], [1,2,2,2,1,1,2], [1,2,2,2,1,2,1], [1,2,2,2,1,2,2], [1,2,2,2,1,3,1], [1,2,2,2,1,3,2], [1,2,2,2,1,4,1], [1,2,2,2,1,4,2], [1,2,2,2,2,1,1], [1,2,2,2,2,1,2], [1,2,2,2,2,2,1], [1,2,2,2,2,2,2], [1,2,2,2,2,3,1], [1,2,2,2,2,3,2], [1,2,2,2,2,4,1], [1,2,2,2,2,4,2], [1,2,2,2,3,1,1], [1,2,2,2,3,1,2], [1,2,2,2,3,2,1], [1,2,2,2,3,2,2], [1,2,2,2,3,3,1], [1,2,2,2,3,3,2], [1,2,2,2,3,4,1], [1,2,2,2,3,4,2], [1,2,3,1,1,1,1], [1,2,3,1,1,1,2], [0,2,3,1,1,2,1], [0,2,3,1,1,2,2], [0,2,3,1,1,3,1], [0,2,3,1,1,3,2], [0,2,3,1,1,4,1], [0,2,3,1,1,4,2], [1,2,3,1,2,1,1], [1,2,3,1,2,1,2], [0,2,3,1,2,2,1], [0,2,3,1,2,2,2], [0,2,3,1,2,3,1], [0,2,3,1,2,3,2], [0,2,3,1,2,4,1], [0,2,3,1,2,4,2], [1,2,3,1,3,1,1], [1,2,3,1,3,1,2], [0,2,3,1,3,2,1], [0,2,3,1,3,2,2], [0,2,3,1,3,3,1], [0,2,3,1,3,3,2], [0,2,3,1,3,4,1], [0,2,3,1,3,4,2], [1,2,3,2,1,1,1], [1,2,3,2,1,1,2], [0,2,3,2,1,2,1], [0,2,3,2,1,2,2], [0,2,3,2,1,3,1], [0,2,3,2,1,3,2], [0,2,3,2,1,4,1], [0,2,3,2,1,4,2], [1,2,3,2,2,1,1], [1,2,3,2,2,1,2], [0,2,3,2,2,2,1], [0,2,3,2,2,2,2], [0,2,3,2,2,3,1], [0,2,3,2,2,3,2], [0,2,3,2,2,4,1], [0,2,3,2,2,4,2], [1,2,3,2,3,1,1], [1,2,3,2,3,1,2], [0,2,3,2,3,2,1], [0,2,3,2,3,2,2], [0,2,3,2,3,3,1], [0,2,3,2,3,3,2], [0,2,3,2,3,4,1], [0,2,3,2,3,4,2], [1,3,1,1,1,1,1], [1,3,1,1,1,1,2], [0,3,1,1,1,2,1], [0,3,1,1,1,2,2], [0,3,1,1,1,3,1], [0,3,1,1,1,3,2], [0,3,1,1,1,4,1], [0,3,1,1,1,4,2], [1,3,1,1,2,1,1], [1,3,1,1,2,1,2], [0,3,1,1,2,2,1], [0,3,1,1,2,2,2], [0,3,1,1,2,3,1], [0,3,1,1,2,3,2], [0,3,1,1,2,4,1], [0,3,1,1,2,4,2], [1,3,1,1,3,1,1], [1,3,1,1,3,1,2], [0,3,1,1,3,2,1], [0,3,1,1,3,2,2], [0,3,1,1,3,3,1], [0,3,1,1,3,3,2], [0,3,1,1,3,4,1], [0,3,1,1,3,4,2], [1,3,1,2,1,1,1], [1,3,1,2,1,1,2], [0,3,1,2,1,2,1], [0,3,1,2,1,2,2], [0,3,1,2,1,3,1], [0,3,1,2,1,3,2], [0,3,1,2,1,4,1], [0,3,1,2,1,4,2], [1,3,1,2,2,1,1], [1,3,1,2,2,1,2], [0,3,1,2,2,2,1], [0,3,1,2,2,2,2], [0,3,1,2,2,3,1], [0,3,1,2,2,3,2], [0,3,1,2,2,4,1], [0,3,1,2,2,4,2], [1,3,1,2,3,1,1], [1,3,1,2,3,1,2], [0,3,1,2,3,2,1], [0,3,1,2,3,2,2], [0,3,1,2,3,3,1], [0,3,1,2,3,3,2], [0,3,1,2,3,4,1], [0,3,1,2,3,4,2], [1,3,2,1,1,1,1], [1,3,2,1,1,1,2], [0,3,2,1,1,2,1], [0,3,2,1,1,2,2], [0,3,2,1,1,3,1], [0,3,2,1,1,3,2], [0,3,2,1,1,4,1], [0,3,2,1,1,4,2], [1,3,2,1,2,1,1], [1,3,2,1,2,1,2], [0,3,2,1,2,2,1], [0,3,2,1,2,2,2], [0,3,2,1,2,3,1], [0,3,2,1,2,3,2], [0,3,2,1,2,4,1], [0,3,2,1,2,4,2], [1,3,2,1,3,1,1], [1,3,2,1,3,1,2], [0,3,2,1,3,2,1], [0,3,2,1,3,2,2], [0,3,2,1,3,3,1], [0,3,2,1,3,3,2], [0,3,2,1,3,4,1], [0,3,2,1,3,4,2], [1,3,2,2,1,1,1], [1,3,2,2,1,1,2], [0,3,2,2,1,2,1], [0,3,2,2,1,2,2], [0,3,2,2,1,3,1], [0,3,2,2,1,3,2], [0,3,2,2,1,4,1], [0,3,2,2,1,4,2], [1,3,2,2,2,1,1], [1,3,2,2,2,1,2], [0,3,2,2,2,2,1], [0,3,2,2,2,2,2], [0,3,2,2,2,3,1], [0,3,2,2,2,3,2], [0,3,2,2,2,4,1], [0,3,2,2,2,4,2], [1,3,2,2,3,1,1], [1,3,2,2,3,1,2], [0,3,2,2,3,2,1], [0,3,2,2,3,2,2], [0,3,2,2,3,3,1], [0,3,2,2,3,3,2], [0,3,2,2,3,4,1], [0,3,2,2,3,4,2], [1,3,3,1,1,1,1], [1,3,3,1,1,1,2], [1,3,3,1,1,2,1], [1,3,3,1,1,2,2], [1,3,3,1,1,3,1], [1,3,3,1,1,3,2], [1,3,3,1,1,4,1], [1,3,3,1,1,4,2], [1,3,3,1,2,1,1], [1,3,3,1,2,1,2], [1,3,3,1,2,2,1], [1,3,3,1,2,2,2], [1,3,3,1,2,3,1], [1,3,3,1,2,3,2], [1,3,3,1,2,4,1], [1,3,3,1,2,4,2], [1,3,3,1,3,1,1], [1,3,3,1,3,1,2], [1,3,3,1,3,2,1], [1,3,3,1,3,2,2], [1,3,3,1,3,3,1], [1,3,3,1,3,3,2], [1,3,3,1,3,4,1], [1,3,3,1,3,4,2], [1,3,3,2,1,1,1], [1,3,3,2,1,1,2], [1,3,3,2,1,2,1], [1,3,3,2,1,2,2], [1,3,3,2,1,3,1], [1,3,3,2,1,3,2], [1,3,3,2,1,4,1], [1,3,3,2,1,4,2], [1,3,3,2,2,1,1], [1,3,3,2,2,1,2], [1,3,3,2,2,2,1], [1,3,3,2,2,2,2], [1,3,3,2,2,3,1], [1,3,3,2,2,3,2], [1,3,3,2,2,4,1], [1,3,3,2,2,4,2], [1,3,3,2,3,1,1], [1,3,3,2,3,1,2], [1,3,3,2,3,2,1], [1,3,3,2,3,2,2], [1,3,3,2,3,3,1], [1,3,3,2,3,3,2], [1,3,3,2,3,4,1], [1,3,3,2,3,4,2]]

monk1_training_set = OneHotEncoder.encode_int_matrix ([el[1:] for el in monk1_train])
monk1_training_labels = [[el[0]] for el in monk1_train]

monk1_test_set = OneHotEncoder.encode_int_matrix ([el[1:] for el in monk1_test])
monk1_test_labels = [[el[0]] for el in monk1_test]

monk2_train = [[0,1,1,1,1,2,2,],[0,1,1,1,1,4,1,],[0,1,1,1,2,1,1,],[0,1,1,1,2,1,2,],[0,1,1,1,2,2,1,],[0,1,1,1,2,3,1,],[0,1,1,1,2,4,1,],[0,1,1,1,3,2,1,],[0,1,1,1,3,4,1,],[0,1,1,2,1,1,1,],[0,1,1,2,1,1,2,],[0,1,1,2,2,3,1,],[0,1,1,2,2,4,1,],[1,1,1,2,2,4,2,],[0,1,1,2,3,1,2,],[1,1,1,2,3,2,2,],[0,1,2,1,1,1,2,],[0,1,2,1,2,1,2,],[1,1,2,1,2,2,2,],[0,1,2,1,2,3,1,],[1,1,2,1,2,3,2,],[0,1,2,1,2,4,1,],[0,1,2,1,3,1,1,],[0,1,2,1,3,1,2,],[1,1,2,1,3,2,2,],[0,1,2,1,3,3,1,],[1,1,2,1,3,3,2,],[0,1,2,1,3,4,1,],[1,1,2,1,3,4,2,],[0,1,2,2,1,2,1,],[0,1,2,2,1,4,1,],[1,1,2,2,2,3,1,],[1,1,2,2,2,4,1,],[0,1,2,2,3,1,1,],[1,1,2,2,3,1,2,],[1,1,2,2,3,3,1,],[0,1,2,2,3,3,2,],[1,1,2,2,3,4,1,],[0,1,2,2,3,4,2,],[0,1,3,1,1,1,2,],[0,1,3,1,1,2,2,],[0,1,3,1,1,3,1,],[0,1,3,1,1,3,2,],[0,1,3,1,2,2,1,],[1,1,3,1,2,2,2,],[1,1,3,1,2,3,2,],[0,1,3,1,2,4,1,],[1,1,3,1,3,2,2,],[0,1,3,1,3,3,1,],[1,1,3,1,3,4,2,],[0,1,3,2,1,3,1,],[1,1,3,2,1,3,2,],[0,1,3,2,1,4,1,],[1,1,3,2,2,1,2,],[0,1,3,2,2,3,2,],[0,1,3,2,2,4,2,],[1,1,3,2,3,2,1,],[0,2,1,1,1,1,1,],[0,2,1,1,1,2,2,],[0,2,1,1,1,3,1,],[1,2,1,1,2,2,2,],[0,2,1,1,3,1,2,],[1,2,1,1,3,2,2,],[1,2,1,1,3,3,2,],[0,2,1,1,3,4,1,],[0,2,1,2,1,1,1,],[1,2,1,2,1,2,2,],[0,2,1,2,1,4,1,],[1,2,1,2,2,2,1,],[0,2,1,2,2,4,2,],[0,2,1,2,3,1,1,],[1,2,1,2,3,1,2,],[0,2,1,2,3,2,2,],[0,2,1,2,3,3,2,],[0,2,1,2,3,4,2,],[0,2,2,1,1,3,1,],[1,2,2,1,1,4,2,],[0,2,2,1,2,1,1,],[1,2,2,1,2,3,1,],[1,2,2,1,3,3,1,],[0,2,2,1,3,3,2,],[1,2,2,1,3,4,1,],[0,2,2,2,1,1,1,],[0,2,2,2,1,2,2,],[0,2,2,2,1,3,2,],[1,2,2,2,1,4,1,],[0,2,2,2,1,4,2,],[1,2,2,2,2,1,1,],[0,2,2,2,2,2,2,],[0,2,2,2,2,3,1,],[1,2,2,2,3,1,1,],[0,2,2,2,3,2,1,],[0,2,2,2,3,2,2,],[0,2,2,2,3,4,2,],[0,2,3,1,1,1,1,],[0,2,3,1,1,1,2,],[1,2,3,1,1,3,2,],[0,2,3,1,2,1,1,],[1,2,3,1,2,3,1,],[0,2,3,1,2,3,2,],[0,2,3,1,2,4,2,],[1,2,3,1,3,1,2,],[1,2,3,1,3,2,1,],[1,2,3,1,3,4,1,],[1,2,3,2,1,1,2,],[1,2,3,2,1,2,1,],[1,2,3,2,1,3,1,],[0,2,3,2,1,4,2,],[1,2,3,2,2,1,1,],[0,2,3,2,2,2,1,],[0,2,3,2,2,3,2,],[0,2,3,2,3,3,1,],[0,2,3,2,3,3,2,],[0,2,3,2,3,4,2,],[0,3,1,1,1,4,1,],[0,3,1,1,2,1,2,],[1,3,1,1,2,2,2,],[1,3,1,1,2,3,2,],[0,3,1,1,2,4,1,],[1,3,1,1,2,4,2,],[0,3,1,1,3,1,1,],[0,3,1,1,3,1,2,],[1,3,1,1,3,2,2,],[1,3,1,1,3,3,2,],[0,3,1,2,1,1,1,],[1,3,1,2,1,2,2,],[0,3,1,2,1,3,1,],[1,3,1,2,1,3,2,],[0,3,1,2,1,4,1,],[1,3,1,2,1,4,2,],[1,3,1,2,2,2,1,],[1,3,1,2,3,1,2,],[1,3,1,2,3,2,1,],[0,3,1,2,3,2,2,],[0,3,1,2,3,4,2,],[0,3,2,1,1,1,2,],[1,3,2,1,1,2,2,],[0,3,2,1,1,3,1,],[1,3,2,1,1,3,2,],[1,3,2,1,2,1,2,],[1,3,2,1,2,2,1,],[0,3,2,1,3,1,1,],[1,3,2,1,3,2,1,],[1,3,2,1,3,3,1,],[0,3,2,1,3,3,2,],[0,3,2,2,1,1,1,],[0,3,2,2,1,2,2,],[1,3,2,2,1,3,1,],[0,3,2,2,1,3,2,],[1,3,2,2,2,1,1,],[0,3,2,2,2,2,1,],[0,3,2,2,2,2,2,],[0,3,2,2,2,3,2,],[1,3,2,2,3,1,1,],[0,3,2,2,3,3,2,],[0,3,2,2,3,4,2,],[0,3,3,1,1,1,1,],[0,3,3,1,1,2,1,],[0,3,3,1,1,3,1,],[1,3,3,1,1,3,2,],[0,3,3,1,2,3,2,],[0,3,3,2,1,1,1,],[1,3,3,2,2,1,1,],[0,3,3,2,2,2,1,],[0,3,3,2,2,3,1,],[0,3,3,2,2,3,2,],[1,3,3,2,3,1,1,],[0,3,3,2,3,2,1,],[0,3,3,2,3,4,2,],]
monk2_test = [[0,1,1,1,1,1,1,],[0,1,1,1,1,1,2,],[0,1,1,1,1,2,1,],[0,1,1,1,1,2,2,],[0,1,1,1,1,3,1,],[0,1,1,1,1,3,2,],[0,1,1,1,1,4,1,],[0,1,1,1,1,4,2,],[0,1,1,1,2,1,1,],[0,1,1,1,2,1,2,],[0,1,1,1,2,2,1,],[0,1,1,1,2,2,2,],[0,1,1,1,2,3,1,],[0,1,1,1,2,3,2,],[0,1,1,1,2,4,1,],[0,1,1,1,2,4,2,],[0,1,1,1,3,1,1,],[0,1,1,1,3,1,2,],[0,1,1,1,3,2,1,],[0,1,1,1,3,2,2,],[0,1,1,1,3,3,1,],[0,1,1,1,3,3,2,],[0,1,1,1,3,4,1,],[0,1,1,1,3,4,2,],[0,1,1,2,1,1,1,],[0,1,1,2,1,1,2,],[0,1,1,2,1,2,1,],[0,1,1,2,1,2,2,],[0,1,1,2,1,3,1,],[0,1,1,2,1,3,2,],[0,1,1,2,1,4,1,],[0,1,1,2,1,4,2,],[0,1,1,2,2,1,1,],[0,1,1,2,2,1,2,],[0,1,1,2,2,2,1,],[1,1,1,2,2,2,2,],[0,1,1,2,2,3,1,],[1,1,1,2,2,3,2,],[0,1,1,2,2,4,1,],[1,1,1,2,2,4,2,],[0,1,1,2,3,1,1,],[0,1,1,2,3,1,2,],[0,1,1,2,3,2,1,],[1,1,1,2,3,2,2,],[0,1,1,2,3,3,1,],[1,1,1,2,3,3,2,],[0,1,1,2,3,4,1,],[1,1,1,2,3,4,2,],[0,1,2,1,1,1,1,],[0,1,2,1,1,1,2,],[0,1,2,1,1,2,1,],[0,1,2,1,1,2,2,],[0,1,2,1,1,3,1,],[0,1,2,1,1,3,2,],[0,1,2,1,1,4,1,],[0,1,2,1,1,4,2,],[0,1,2,1,2,1,1,],[0,1,2,1,2,1,2,],[0,1,2,1,2,2,1,],[1,1,2,1,2,2,2,],[0,1,2,1,2,3,1,],[1,1,2,1,2,3,2,],[0,1,2,1,2,4,1,],[1,1,2,1,2,4,2,],[0,1,2,1,3,1,1,],[0,1,2,1,3,1,2,],[0,1,2,1,3,2,1,],[1,1,2,1,3,2,2,],[0,1,2,1,3,3,1,],[1,1,2,1,3,3,2,],[0,1,2,1,3,4,1,],[1,1,2,1,3,4,2,],[0,1,2,2,1,1,1,],[0,1,2,2,1,1,2,],[0,1,2,2,1,2,1,],[1,1,2,2,1,2,2,],[0,1,2,2,1,3,1,],[1,1,2,2,1,3,2,],[0,1,2,2,1,4,1,],[1,1,2,2,1,4,2,],[0,1,2,2,2,1,1,],[1,1,2,2,2,1,2,],[1,1,2,2,2,2,1,],[0,1,2,2,2,2,2,],[1,1,2,2,2,3,1,],[0,1,2,2,2,3,2,],[1,1,2,2,2,4,1,],[0,1,2,2,2,4,2,],[0,1,2,2,3,1,1,],[1,1,2,2,3,1,2,],[1,1,2,2,3,2,1,],[0,1,2,2,3,2,2,],[1,1,2,2,3,3,1,],[0,1,2,2,3,3,2,],[1,1,2,2,3,4,1,],[0,1,2,2,3,4,2,],[0,1,3,1,1,1,1,],[0,1,3,1,1,1,2,],[0,1,3,1,1,2,1,],[0,1,3,1,1,2,2,],[0,1,3,1,1,3,1,],[0,1,3,1,1,3,2,],[0,1,3,1,1,4,1,],[0,1,3,1,1,4,2,],[0,1,3,1,2,1,1,],[0,1,3,1,2,1,2,],[0,1,3,1,2,2,1,],[1,1,3,1,2,2,2,],[0,1,3,1,2,3,1,],[1,1,3,1,2,3,2,],[0,1,3,1,2,4,1,],[1,1,3,1,2,4,2,],[0,1,3,1,3,1,1,],[0,1,3,1,3,1,2,],[0,1,3,1,3,2,1,],[1,1,3,1,3,2,2,],[0,1,3,1,3,3,1,],[1,1,3,1,3,3,2,],[0,1,3,1,3,4,1,],[1,1,3,1,3,4,2,],[0,1,3,2,1,1,1,],[0,1,3,2,1,1,2,],[0,1,3,2,1,2,1,],[1,1,3,2,1,2,2,],[0,1,3,2,1,3,1,],[1,1,3,2,1,3,2,],[0,1,3,2,1,4,1,],[1,1,3,2,1,4,2,],[0,1,3,2,2,1,1,],[1,1,3,2,2,1,2,],[1,1,3,2,2,2,1,],[0,1,3,2,2,2,2,],[1,1,3,2,2,3,1,],[0,1,3,2,2,3,2,],[1,1,3,2,2,4,1,],[0,1,3,2,2,4,2,],[0,1,3,2,3,1,1,],[1,1,3,2,3,1,2,],[1,1,3,2,3,2,1,],[0,1,3,2,3,2,2,],[1,1,3,2,3,3,1,],[0,1,3,2,3,3,2,],[1,1,3,2,3,4,1,],[0,1,3,2,3,4,2,],[0,2,1,1,1,1,1,],[0,2,1,1,1,1,2,],[0,2,1,1,1,2,1,],[0,2,1,1,1,2,2,],[0,2,1,1,1,3,1,],[0,2,1,1,1,3,2,],[0,2,1,1,1,4,1,],[0,2,1,1,1,4,2,],[0,2,1,1,2,1,1,],[0,2,1,1,2,1,2,],[0,2,1,1,2,2,1,],[1,2,1,1,2,2,2,],[0,2,1,1,2,3,1,],[1,2,1,1,2,3,2,],[0,2,1,1,2,4,1,],[1,2,1,1,2,4,2,],[0,2,1,1,3,1,1,],[0,2,1,1,3,1,2,],[0,2,1,1,3,2,1,],[1,2,1,1,3,2,2,],[0,2,1,1,3,3,1,],[1,2,1,1,3,3,2,],[0,2,1,1,3,4,1,],[1,2,1,1,3,4,2,],[0,2,1,2,1,1,1,],[0,2,1,2,1,1,2,],[0,2,1,2,1,2,1,],[1,2,1,2,1,2,2,],[0,2,1,2,1,3,1,],[1,2,1,2,1,3,2,],[0,2,1,2,1,4,1,],[1,2,1,2,1,4,2,],[0,2,1,2,2,1,1,],[1,2,1,2,2,1,2,],[1,2,1,2,2,2,1,],[0,2,1,2,2,2,2,],[1,2,1,2,2,3,1,],[0,2,1,2,2,3,2,],[1,2,1,2,2,4,1,],[0,2,1,2,2,4,2,],[0,2,1,2,3,1,1,],[1,2,1,2,3,1,2,],[1,2,1,2,3,2,1,],[0,2,1,2,3,2,2,],[1,2,1,2,3,3,1,],[0,2,1,2,3,3,2,],[1,2,1,2,3,4,1,],[0,2,1,2,3,4,2,],[0,2,2,1,1,1,1,],[0,2,2,1,1,1,2,],[0,2,2,1,1,2,1,],[1,2,2,1,1,2,2,],[0,2,2,1,1,3,1,],[1,2,2,1,1,3,2,],[0,2,2,1,1,4,1,],[1,2,2,1,1,4,2,],[0,2,2,1,2,1,1,],[1,2,2,1,2,1,2,],[1,2,2,1,2,2,1,],[0,2,2,1,2,2,2,],[1,2,2,1,2,3,1,],[0,2,2,1,2,3,2,],[1,2,2,1,2,4,1,],[0,2,2,1,2,4,2,],[0,2,2,1,3,1,1,],[1,2,2,1,3,1,2,],[1,2,2,1,3,2,1,],[0,2,2,1,3,2,2,],[1,2,2,1,3,3,1,],[0,2,2,1,3,3,2,],[1,2,2,1,3,4,1,],[0,2,2,1,3,4,2,],[0,2,2,2,1,1,1,],[1,2,2,2,1,1,2,],[1,2,2,2,1,2,1,],[0,2,2,2,1,2,2,],[1,2,2,2,1,3,1,],[0,2,2,2,1,3,2,],[1,2,2,2,1,4,1,],[0,2,2,2,1,4,2,],[1,2,2,2,2,1,1,],[0,2,2,2,2,1,2,],[0,2,2,2,2,2,1,],[0,2,2,2,2,2,2,],[0,2,2,2,2,3,1,],[0,2,2,2,2,3,2,],[0,2,2,2,2,4,1,],[0,2,2,2,2,4,2,],[1,2,2,2,3,1,1,],[0,2,2,2,3,1,2,],[0,2,2,2,3,2,1,],[0,2,2,2,3,2,2,],[0,2,2,2,3,3,1,],[0,2,2,2,3,3,2,],[0,2,2,2,3,4,1,],[0,2,2,2,3,4,2,],[0,2,3,1,1,1,1,],[0,2,3,1,1,1,2,],[0,2,3,1,1,2,1,],[1,2,3,1,1,2,2,],[0,2,3,1,1,3,1,],[1,2,3,1,1,3,2,],[0,2,3,1,1,4,1,],[1,2,3,1,1,4,2,],[0,2,3,1,2,1,1,],[1,2,3,1,2,1,2,],[1,2,3,1,2,2,1,],[0,2,3,1,2,2,2,],[1,2,3,1,2,3,1,],[0,2,3,1,2,3,2,],[1,2,3,1,2,4,1,],[0,2,3,1,2,4,2,],[0,2,3,1,3,1,1,],[1,2,3,1,3,1,2,],[1,2,3,1,3,2,1,],[0,2,3,1,3,2,2,],[1,2,3,1,3,3,1,],[0,2,3,1,3,3,2,],[1,2,3,1,3,4,1,],[0,2,3,1,3,4,2,],[0,2,3,2,1,1,1,],[1,2,3,2,1,1,2,],[1,2,3,2,1,2,1,],[0,2,3,2,1,2,2,],[1,2,3,2,1,3,1,],[0,2,3,2,1,3,2,],[1,2,3,2,1,4,1,],[0,2,3,2,1,4,2,],[1,2,3,2,2,1,1,],[0,2,3,2,2,1,2,],[0,2,3,2,2,2,1,],[0,2,3,2,2,2,2,],[0,2,3,2,2,3,1,],[0,2,3,2,2,3,2,],[0,2,3,2,2,4,1,],[0,2,3,2,2,4,2,],[1,2,3,2,3,1,1,],[0,2,3,2,3,1,2,],[0,2,3,2,3,2,1,],[0,2,3,2,3,2,2,],[0,2,3,2,3,3,1,],[0,2,3,2,3,3,2,],[0,2,3,2,3,4,1,],[0,2,3,2,3,4,2,],[0,3,1,1,1,1,1,],[0,3,1,1,1,1,2,],[0,3,1,1,1,2,1,],[0,3,1,1,1,2,2,],[0,3,1,1,1,3,1,],[0,3,1,1,1,3,2,],[0,3,1,1,1,4,1,],[0,3,1,1,1,4,2,],[0,3,1,1,2,1,1,],[0,3,1,1,2,1,2,],[0,3,1,1,2,2,1,],[1,3,1,1,2,2,2,],[0,3,1,1,2,3,1,],[1,3,1,1,2,3,2,],[0,3,1,1,2,4,1,],[1,3,1,1,2,4,2,],[0,3,1,1,3,1,1,],[0,3,1,1,3,1,2,],[0,3,1,1,3,2,1,],[1,3,1,1,3,2,2,],[0,3,1,1,3,3,1,],[1,3,1,1,3,3,2,],[0,3,1,1,3,4,1,],[1,3,1,1,3,4,2,],[0,3,1,2,1,1,1,],[0,3,1,2,1,1,2,],[0,3,1,2,1,2,1,],[1,3,1,2,1,2,2,],[0,3,1,2,1,3,1,],[1,3,1,2,1,3,2,],[0,3,1,2,1,4,1,],[1,3,1,2,1,4,2,],[0,3,1,2,2,1,1,],[1,3,1,2,2,1,2,],[1,3,1,2,2,2,1,],[0,3,1,2,2,2,2,],[1,3,1,2,2,3,1,],[0,3,1,2,2,3,2,],[1,3,1,2,2,4,1,],[0,3,1,2,2,4,2,],[0,3,1,2,3,1,1,],[1,3,1,2,3,1,2,],[1,3,1,2,3,2,1,],[0,3,1,2,3,2,2,],[1,3,1,2,3,3,1,],[0,3,1,2,3,3,2,],[1,3,1,2,3,4,1,],[0,3,1,2,3,4,2,],[0,3,2,1,1,1,1,],[0,3,2,1,1,1,2,],[0,3,2,1,1,2,1,],[1,3,2,1,1,2,2,],[0,3,2,1,1,3,1,],[1,3,2,1,1,3,2,],[0,3,2,1,1,4,1,],[1,3,2,1,1,4,2,],[0,3,2,1,2,1,1,],[1,3,2,1,2,1,2,],[1,3,2,1,2,2,1,],[0,3,2,1,2,2,2,],[1,3,2,1,2,3,1,],[0,3,2,1,2,3,2,],[1,3,2,1,2,4,1,],[0,3,2,1,2,4,2,],[0,3,2,1,3,1,1,],[1,3,2,1,3,1,2,],[1,3,2,1,3,2,1,],[0,3,2,1,3,2,2,],[1,3,2,1,3,3,1,],[0,3,2,1,3,3,2,],[1,3,2,1,3,4,1,],[0,3,2,1,3,4,2,],[0,3,2,2,1,1,1,],[1,3,2,2,1,1,2,],[1,3,2,2,1,2,1,],[0,3,2,2,1,2,2,],[1,3,2,2,1,3,1,],[0,3,2,2,1,3,2,],[1,3,2,2,1,4,1,],[0,3,2,2,1,4,2,],[1,3,2,2,2,1,1,],[0,3,2,2,2,1,2,],[0,3,2,2,2,2,1,],[0,3,2,2,2,2,2,],[0,3,2,2,2,3,1,],[0,3,2,2,2,3,2,],[0,3,2,2,2,4,1,],[0,3,2,2,2,4,2,],[1,3,2,2,3,1,1,],[0,3,2,2,3,1,2,],[0,3,2,2,3,2,1,],[0,3,2,2,3,2,2,],[0,3,2,2,3,3,1,],[0,3,2,2,3,3,2,],[0,3,2,2,3,4,1,],[0,3,2,2,3,4,2,],[0,3,3,1,1,1,1,],[0,3,3,1,1,1,2,],[0,3,3,1,1,2,1,],[1,3,3,1,1,2,2,],[0,3,3,1,1,3,1,],[1,3,3,1,1,3,2,],[0,3,3,1,1,4,1,],[1,3,3,1,1,4,2,],[0,3,3,1,2,1,1,],[1,3,3,1,2,1,2,],[1,3,3,1,2,2,1,],[0,3,3,1,2,2,2,],[1,3,3,1,2,3,1,],[0,3,3,1,2,3,2,],[1,3,3,1,2,4,1,],[0,3,3,1,2,4,2,],[0,3,3,1,3,1,1,],[1,3,3,1,3,1,2,],[1,3,3,1,3,2,1,],[0,3,3,1,3,2,2,],[1,3,3,1,3,3,1,],[0,3,3,1,3,3,2,],[1,3,3,1,3,4,1,],[0,3,3,1,3,4,2,],[0,3,3,2,1,1,1,],[1,3,3,2,1,1,2,],[1,3,3,2,1,2,1,],[0,3,3,2,1,2,2,],[1,3,3,2,1,3,1,],[0,3,3,2,1,3,2,],[1,3,3,2,1,4,1,],[0,3,3,2,1,4,2,],[1,3,3,2,2,1,1,],[0,3,3,2,2,1,2,],[0,3,3,2,2,2,1,],[0,3,3,2,2,2,2,],[0,3,3,2,2,3,1,],[0,3,3,2,2,3,2,],[0,3,3,2,2,4,1,],[0,3,3,2,2,4,2,],[1,3,3,2,3,1,1,],[0,3,3,2,3,1,2,],[0,3,3,2,3,2,1,],[0,3,3,2,3,2,2,],[0,3,3,2,3,3,1,],[0,3,3,2,3,3,2,],[0,3,3,2,3,4,1,],[0,3,3,2,3,4,2,],]

monk2_training_set = OneHotEncoder.encode_int_matrix ([el[1:] for el in monk2_train])
monk2_training_labels = [[el[0]] for el in monk2_train]

monk2_test_set = OneHotEncoder.encode_int_matrix ([el[1:] for el in monk2_test])
monk2_test_labels = [[el[0]] for el in monk2_test]

monk3_train = [[1,1,1,1,1,1,2],[1,1,1,1,1,2,1],[1,1,1,1,1,2,2],[0,1,1,1,1,3,1],[0,1,1,1,1,4,1],[1,1,1,1,2,1,1],[1,1,1,1,2,2,2],[0,1,1,1,2,4,2],[1,1,1,2,1,2,2],[0,1,1,2,1,4,2],[1,1,1,2,2,2,2],[0,1,1,2,2,4,1],[0,1,1,2,2,4,2],[1,1,1,2,3,1,1],[1,1,1,2,3,1,2],[1,1,1,2,3,3,1],[1,1,1,2,3,3,2],[1,1,2,1,1,3,1],[1,1,2,1,2,2,1],[1,1,2,1,2,2,2],[0,1,2,1,2,3,1],[1,1,2,1,3,1,1],[1,1,2,1,3,1,2],[1,1,2,1,3,2,1],[1,1,2,1,3,2,2],[1,1,2,1,3,3,2],[0,1,2,1,3,4,1],[1,1,2,2,1,3,1],[0,1,2,2,1,4,2],[1,1,2,2,2,1,1],[1,1,2,2,2,2,1],[1,1,2,2,2,2,2],[1,1,2,2,3,1,1],[1,1,2,2,3,2,1],[1,1,2,2,3,2,2],[0,1,3,1,1,2,1],[0,1,3,1,1,4,1],[0,1,3,1,2,3,2],[0,1,3,1,2,4,1],[0,1,3,1,3,1,1],[0,1,3,1,3,3,1],[0,1,3,2,1,1,1],[0,1,3,2,1,1,2],[0,1,3,2,1,2,1],[0,1,3,2,1,4,2],[0,1,3,2,2,3,2],[0,1,3,2,2,4,2],[0,1,3,2,3,4,1],[1,2,1,1,1,1,1],[1,2,1,1,1,1,2],[0,2,1,1,1,4,1],[0,2,1,1,1,4,2],[1,2,1,1,2,1,1],[1,2,1,1,2,1,2],[1,2,1,1,3,2,2],[1,2,1,1,3,3,2],[0,2,1,1,3,4,1],[1,2,1,2,1,2,2],[0,2,1,2,2,4,1],[1,2,1,2,3,1,2],[1,2,2,1,1,3,2],[0,2,2,1,1,4,2],[1,2,2,1,2,1,2],[0,2,2,1,2,2,1],[1,2,2,1,3,1,1],[1,2,2,1,3,2,2],[0,2,2,1,3,3,1],[0,2,2,1,3,3,2],[0,2,2,1,3,4,2],[1,2,2,2,1,2,2],[1,2,2,2,2,1,2],[1,2,2,2,2,3,1],[1,2,2,2,2,3,2],[0,2,2,2,3,4,1],[1,2,3,1,1,3,1],[0,2,3,1,2,1,1],[0,2,3,1,2,2,1],[0,2,3,1,2,2,2],[0,2,3,1,2,3,2],[0,2,3,1,3,3,1],[0,2,3,2,1,1,2],[0,2,3,2,1,2,2],[0,2,3,2,1,4,1],[0,2,3,2,2,3,1],[0,2,3,2,2,4,2],[0,2,3,2,3,1,1],[0,2,3,2,3,2,1],[0,2,3,2,3,4,2],[1,3,1,1,1,1,1],[1,3,1,1,1,2,1],[1,3,1,1,1,3,1],[0,3,1,1,2,4,2],[1,3,1,1,3,1,2],[0,3,1,1,3,4,2],[1,3,1,2,1,2,1],[1,3,1,2,2,3,2],[0,3,1,2,2,4,2],[1,3,1,2,3,1,1],[1,3,2,1,1,2,2],[0,3,2,1,1,4,1],[1,3,2,1,2,3,1],[1,3,2,1,3,1,2],[1,3,2,2,1,2,2],[1,3,2,2,1,3,2],[1,3,2,2,2,1,2],[1,3,2,2,3,1,1],[1,3,2,2,3,3,2],[0,3,2,2,3,4,1],[1,3,3,1,1,3,2],[1,3,3,1,1,4,1],[0,3,3,1,2,4,2],[0,3,3,1,3,1,1],[0,3,3,1,3,2,1],[0,3,3,1,3,2,2],[0,3,3,1,3,4,1],[0,3,3,2,1,1,1],[0,3,3,2,1,1,2],[0,3,3,2,2,2,2],[0,3,3,2,2,3,2],[0,3,3,2,3,1,1],[0,3,3,2,3,3,2],[0,3,3,2,3,4,2],]
monk3_test = [[1,1,1,1,1,1,1],[1,1,1,1,1,1,2],[1,1,1,1,1,2,1],[1,1,1,1,1,2,2],[1,1,1,1,1,3,1],[1,1,1,1,1,3,2],[0,1,1,1,1,4,1],[0,1,1,1,1,4,2],[1,1,1,1,2,1,1],[1,1,1,1,2,1,2],[1,1,1,1,2,2,1],[1,1,1,1,2,2,2],[1,1,1,1,2,3,1],[1,1,1,1,2,3,2],[0,1,1,1,2,4,1],[0,1,1,1,2,4,2],[1,1,1,1,3,1,1],[1,1,1,1,3,1,2],[1,1,1,1,3,2,1],[1,1,1,1,3,2,2],[1,1,1,1,3,3,1],[1,1,1,1,3,3,2],[0,1,1,1,3,4,1],[0,1,1,1,3,4,2],[1,1,1,2,1,1,1],[1,1,1,2,1,1,2],[1,1,1,2,1,2,1],[1,1,1,2,1,2,2],[1,1,1,2,1,3,1],[1,1,1,2,1,3,2],[0,1,1,2,1,4,1],[0,1,1,2,1,4,2],[1,1,1,2,2,1,1],[1,1,1,2,2,1,2],[1,1,1,2,2,2,1],[1,1,1,2,2,2,2],[1,1,1,2,2,3,1],[1,1,1,2,2,3,2],[0,1,1,2,2,4,1],[0,1,1,2,2,4,2],[1,1,1,2,3,1,1],[1,1,1,2,3,1,2],[1,1,1,2,3,2,1],[1,1,1,2,3,2,2],[1,1,1,2,3,3,1],[1,1,1,2,3,3,2],[0,1,1,2,3,4,1],[0,1,1,2,3,4,2],[1,1,2,1,1,1,1],[1,1,2,1,1,1,2],[1,1,2,1,1,2,1],[1,1,2,1,1,2,2],[1,1,2,1,1,3,1],[1,1,2,1,1,3,2],[0,1,2,1,1,4,1],[0,1,2,1,1,4,2],[1,1,2,1,2,1,1],[1,1,2,1,2,1,2],[1,1,2,1,2,2,1],[1,1,2,1,2,2,2],[1,1,2,1,2,3,1],[1,1,2,1,2,3,2],[0,1,2,1,2,4,1],[0,1,2,1,2,4,2],[1,1,2,1,3,1,1],[1,1,2,1,3,1,2],[1,1,2,1,3,2,1],[1,1,2,1,3,2,2],[1,1,2,1,3,3,1],[1,1,2,1,3,3,2],[0,1,2,1,3,4,1],[0,1,2,1,3,4,2],[1,1,2,2,1,1,1],[1,1,2,2,1,1,2],[1,1,2,2,1,2,1],[1,1,2,2,1,2,2],[1,1,2,2,1,3,1],[1,1,2,2,1,3,2],[0,1,2,2,1,4,1],[0,1,2,2,1,4,2],[1,1,2,2,2,1,1],[1,1,2,2,2,1,2],[1,1,2,2,2,2,1],[1,1,2,2,2,2,2],[1,1,2,2,2,3,1],[1,1,2,2,2,3,2],[0,1,2,2,2,4,1],[0,1,2,2,2,4,2],[1,1,2,2,3,1,1],[1,1,2,2,3,1,2],[1,1,2,2,3,2,1],[1,1,2,2,3,2,2],[1,1,2,2,3,3,1],[1,1,2,2,3,3,2],[0,1,2,2,3,4,1],[0,1,2,2,3,4,2],[0,1,3,1,1,1,1],[0,1,3,1,1,1,2],[0,1,3,1,1,2,1],[0,1,3,1,1,2,2],[1,1,3,1,1,3,1],[1,1,3,1,1,3,2],[0,1,3,1,1,4,1],[0,1,3,1,1,4,2],[0,1,3,1,2,1,1],[0,1,3,1,2,1,2],[0,1,3,1,2,2,1],[0,1,3,1,2,2,2],[0,1,3,1,2,3,1],[0,1,3,1,2,3,2],[0,1,3,1,2,4,1],[0,1,3,1,2,4,2],[0,1,3,1,3,1,1],[0,1,3,1,3,1,2],[0,1,3,1,3,2,1],[0,1,3,1,3,2,2],[0,1,3,1,3,3,1],[0,1,3,1,3,3,2],[0,1,3,1,3,4,1],[0,1,3,1,3,4,2],[0,1,3,2,1,1,1],[0,1,3,2,1,1,2],[0,1,3,2,1,2,1],[0,1,3,2,1,2,2],[1,1,3,2,1,3,1],[1,1,3,2,1,3,2],[0,1,3,2,1,4,1],[0,1,3,2,1,4,2],[0,1,3,2,2,1,1],[0,1,3,2,2,1,2],[0,1,3,2,2,2,1],[0,1,3,2,2,2,2],[0,1,3,2,2,3,1],[0,1,3,2,2,3,2],[0,1,3,2,2,4,1],[0,1,3,2,2,4,2],[0,1,3,2,3,1,1],[0,1,3,2,3,1,2],[0,1,3,2,3,2,1],[0,1,3,2,3,2,2],[0,1,3,2,3,3,1],[0,1,3,2,3,3,2],[0,1,3,2,3,4,1],[0,1,3,2,3,4,2],[1,2,1,1,1,1,1],[1,2,1,1,1,1,2],[1,2,1,1,1,2,1],[1,2,1,1,1,2,2],[1,2,1,1,1,3,1],[1,2,1,1,1,3,2],[0,2,1,1,1,4,1],[0,2,1,1,1,4,2],[1,2,1,1,2,1,1],[1,2,1,1,2,1,2],[1,2,1,1,2,2,1],[1,2,1,1,2,2,2],[1,2,1,1,2,3,1],[1,2,1,1,2,3,2],[0,2,1,1,2,4,1],[0,2,1,1,2,4,2],[1,2,1,1,3,1,1],[1,2,1,1,3,1,2],[1,2,1,1,3,2,1],[1,2,1,1,3,2,2],[1,2,1,1,3,3,1],[1,2,1,1,3,3,2],[0,2,1,1,3,4,1],[0,2,1,1,3,4,2],[1,2,1,2,1,1,1],[1,2,1,2,1,1,2],[1,2,1,2,1,2,1],[1,2,1,2,1,2,2],[1,2,1,2,1,3,1],[1,2,1,2,1,3,2],[0,2,1,2,1,4,1],[0,2,1,2,1,4,2],[1,2,1,2,2,1,1],[1,2,1,2,2,1,2],[1,2,1,2,2,2,1],[1,2,1,2,2,2,2],[1,2,1,2,2,3,1],[1,2,1,2,2,3,2],[0,2,1,2,2,4,1],[0,2,1,2,2,4,2],[1,2,1,2,3,1,1],[1,2,1,2,3,1,2],[1,2,1,2,3,2,1],[1,2,1,2,3,2,2],[1,2,1,2,3,3,1],[1,2,1,2,3,3,2],[0,2,1,2,3,4,1],[0,2,1,2,3,4,2],[1,2,2,1,1,1,1],[1,2,2,1,1,1,2],[1,2,2,1,1,2,1],[1,2,2,1,1,2,2],[1,2,2,1,1,3,1],[1,2,2,1,1,3,2],[0,2,2,1,1,4,1],[0,2,2,1,1,4,2],[1,2,2,1,2,1,1],[1,2,2,1,2,1,2],[1,2,2,1,2,2,1],[1,2,2,1,2,2,2],[1,2,2,1,2,3,1],[1,2,2,1,2,3,2],[0,2,2,1,2,4,1],[0,2,2,1,2,4,2],[1,2,2,1,3,1,1],[1,2,2,1,3,1,2],[1,2,2,1,3,2,1],[1,2,2,1,3,2,2],[1,2,2,1,3,3,1],[1,2,2,1,3,3,2],[0,2,2,1,3,4,1],[0,2,2,1,3,4,2],[1,2,2,2,1,1,1],[1,2,2,2,1,1,2],[1,2,2,2,1,2,1],[1,2,2,2,1,2,2],[1,2,2,2,1,3,1],[1,2,2,2,1,3,2],[0,2,2,2,1,4,1],[0,2,2,2,1,4,2],[1,2,2,2,2,1,1],[1,2,2,2,2,1,2],[1,2,2,2,2,2,1],[1,2,2,2,2,2,2],[1,2,2,2,2,3,1],[1,2,2,2,2,3,2],[0,2,2,2,2,4,1],[0,2,2,2,2,4,2],[1,2,2,2,3,1,1],[1,2,2,2,3,1,2],[1,2,2,2,3,2,1],[1,2,2,2,3,2,2],[1,2,2,2,3,3,1],[1,2,2,2,3,3,2],[0,2,2,2,3,4,1],[0,2,2,2,3,4,2],[0,2,3,1,1,1,1],[0,2,3,1,1,1,2],[0,2,3,1,1,2,1],[0,2,3,1,1,2,2],[1,2,3,1,1,3,1],[1,2,3,1,1,3,2],[0,2,3,1,1,4,1],[0,2,3,1,1,4,2],[0,2,3,1,2,1,1],[0,2,3,1,2,1,2],[0,2,3,1,2,2,1],[0,2,3,1,2,2,2],[0,2,3,1,2,3,1],[0,2,3,1,2,3,2],[0,2,3,1,2,4,1],[0,2,3,1,2,4,2],[0,2,3,1,3,1,1],[0,2,3,1,3,1,2],[0,2,3,1,3,2,1],[0,2,3,1,3,2,2],[0,2,3,1,3,3,1],[0,2,3,1,3,3,2],[0,2,3,1,3,4,1],[0,2,3,1,3,4,2],[0,2,3,2,1,1,1],[0,2,3,2,1,1,2],[0,2,3,2,1,2,1],[0,2,3,2,1,2,2],[1,2,3,2,1,3,1],[1,2,3,2,1,3,2],[0,2,3,2,1,4,1],[0,2,3,2,1,4,2],[0,2,3,2,2,1,1],[0,2,3,2,2,1,2],[0,2,3,2,2,2,1],[0,2,3,2,2,2,2],[0,2,3,2,2,3,1],[0,2,3,2,2,3,2],[0,2,3,2,2,4,1],[0,2,3,2,2,4,2],[0,2,3,2,3,1,1],[0,2,3,2,3,1,2],[0,2,3,2,3,2,1],[0,2,3,2,3,2,2],[0,2,3,2,3,3,1],[0,2,3,2,3,3,2],[0,2,3,2,3,4,1],[0,2,3,2,3,4,2],[1,3,1,1,1,1,1],[1,3,1,1,1,1,2],[1,3,1,1,1,2,1],[1,3,1,1,1,2,2],[1,3,1,1,1,3,1],[1,3,1,1,1,3,2],[0,3,1,1,1,4,1],[0,3,1,1,1,4,2],[1,3,1,1,2,1,1],[1,3,1,1,2,1,2],[1,3,1,1,2,2,1],[1,3,1,1,2,2,2],[1,3,1,1,2,3,1],[1,3,1,1,2,3,2],[0,3,1,1,2,4,1],[0,3,1,1,2,4,2],[1,3,1,1,3,1,1],[1,3,1,1,3,1,2],[1,3,1,1,3,2,1],[1,3,1,1,3,2,2],[1,3,1,1,3,3,1],[1,3,1,1,3,3,2],[0,3,1,1,3,4,1],[0,3,1,1,3,4,2],[1,3,1,2,1,1,1],[1,3,1,2,1,1,2],[1,3,1,2,1,2,1],[1,3,1,2,1,2,2],[1,3,1,2,1,3,1],[1,3,1,2,1,3,2],[0,3,1,2,1,4,1],[0,3,1,2,1,4,2],[1,3,1,2,2,1,1],[1,3,1,2,2,1,2],[1,3,1,2,2,2,1],[1,3,1,2,2,2,2],[1,3,1,2,2,3,1],[1,3,1,2,2,3,2],[0,3,1,2,2,4,1],[0,3,1,2,2,4,2],[1,3,1,2,3,1,1],[1,3,1,2,3,1,2],[1,3,1,2,3,2,1],[1,3,1,2,3,2,2],[1,3,1,2,3,3,1],[1,3,1,2,3,3,2],[0,3,1,2,3,4,1],[0,3,1,2,3,4,2],[1,3,2,1,1,1,1],[1,3,2,1,1,1,2],[1,3,2,1,1,2,1],[1,3,2,1,1,2,2],[1,3,2,1,1,3,1],[1,3,2,1,1,3,2],[0,3,2,1,1,4,1],[0,3,2,1,1,4,2],[1,3,2,1,2,1,1],[1,3,2,1,2,1,2],[1,3,2,1,2,2,1],[1,3,2,1,2,2,2],[1,3,2,1,2,3,1],[1,3,2,1,2,3,2],[0,3,2,1,2,4,1],[0,3,2,1,2,4,2],[1,3,2,1,3,1,1],[1,3,2,1,3,1,2],[1,3,2,1,3,2,1],[1,3,2,1,3,2,2],[1,3,2,1,3,3,1],[1,3,2,1,3,3,2],[0,3,2,1,3,4,1],[0,3,2,1,3,4,2],[1,3,2,2,1,1,1],[1,3,2,2,1,1,2],[1,3,2,2,1,2,1],[1,3,2,2,1,2,2],[1,3,2,2,1,3,1],[1,3,2,2,1,3,2],[0,3,2,2,1,4,1],[0,3,2,2,1,4,2],[1,3,2,2,2,1,1],[1,3,2,2,2,1,2],[1,3,2,2,2,2,1],[1,3,2,2,2,2,2],[1,3,2,2,2,3,1],[1,3,2,2,2,3,2],[0,3,2,2,2,4,1],[0,3,2,2,2,4,2],[1,3,2,2,3,1,1],[1,3,2,2,3,1,2],[1,3,2,2,3,2,1],[1,3,2,2,3,2,2],[1,3,2,2,3,3,1],[1,3,2,2,3,3,2],[0,3,2,2,3,4,1],[0,3,2,2,3,4,2],[0,3,3,1,1,1,1],[0,3,3,1,1,1,2],[0,3,3,1,1,2,1],[0,3,3,1,1,2,2],[1,3,3,1,1,3,1],[1,3,3,1,1,3,2],[0,3,3,1,1,4,1],[0,3,3,1,1,4,2],[0,3,3,1,2,1,1],[0,3,3,1,2,1,2],[0,3,3,1,2,2,1],[0,3,3,1,2,2,2],[0,3,3,1,2,3,1],[0,3,3,1,2,3,2],[0,3,3,1,2,4,1],[0,3,3,1,2,4,2],[0,3,3,1,3,1,1],[0,3,3,1,3,1,2],[0,3,3,1,3,2,1],[0,3,3,1,3,2,2],[0,3,3,1,3,3,1],[0,3,3,1,3,3,2],[0,3,3,1,3,4,1],[0,3,3,1,3,4,2],[0,3,3,2,1,1,1],[0,3,3,2,1,1,2],[0,3,3,2,1,2,1],[0,3,3,2,1,2,2],[1,3,3,2,1,3,1],[1,3,3,2,1,3,2],[0,3,3,2,1,4,1],[0,3,3,2,1,4,2],[0,3,3,2,2,1,1],[0,3,3,2,2,1,2],[0,3,3,2,2,2,1],[0,3,3,2,2,2,2],[0,3,3,2,2,3,1],[0,3,3,2,2,3,2],[0,3,3,2,2,4,1],[0,3,3,2,2,4,2],[0,3,3,2,3,1,1],[0,3,3,2,3,1,2],[0,3,3,2,3,2,1],[0,3,3,2,3,2,2],[0,3,3,2,3,3,1],[0,3,3,2,3,3,2],[0,3,3,2,3,4,1],[0,3,3,2,3,4,2],]

monk3_training_set = OneHotEncoder.encode_int_matrix ([el[1:] for el in monk3_train])
monk3_training_labels = [[el[0]] for el in monk3_train]

monk3_test_set = OneHotEncoder.encode_int_matrix ([el[1:] for el in monk3_test])
monk3_test_labels = [[el[0]] for el in monk3_test]

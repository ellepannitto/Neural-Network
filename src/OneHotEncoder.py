'''
  this module implements one-hot encoding of categorical features.
'''

def encode_col (col, col_minimum, col_maximum):
	'''
	
	  encodes a single column of integers.
	  
	  :params:
	   col:         the column of n infegers to be encoded
	   col_minimum: the minimum value that appears in the column
	   col_maximum: the maximum value that appears in the column
      
      :returns: a matrix of shape n * (col_maximum-col_minimum) that contains the one-hot encoding representation of col
	'''
	encoded_matrix = []
	for c in col:
		row = [0]*(col_maximum-col_minimum+1)
		row[c-col_minimum] = 1
		encoded_matrix.append(row)
	return encoded_matrix

def encode_int_matrix ( matrix ):
	'''
	
	  processes an integer matrix, performing one-hot encoding of every column.
	  
	  :params:
	   matrix: the matrix of to be encoded
	  
	  :returns: the encoded matrix
	  
	'''
	
	new_matrix = [[] for _ in matrix]
	for c in range (len(matrix[0])):
		col = [row[c] for row in matrix]
		col_maximum = max( col )
		col_minimum = min( col )
		encoded_col = encode_col (col, col_minimum, col_maximum)
		for i,row in enumerate(encoded_col):
			new_matrix[i].extend(row)
	return new_matrix

# unit tests
if __name__ == "__main__":

	m = [ [1,1,1,1,1,3,1],
		  [1,1,1,1,1,3,2],
		  [1,1,1,1,3,2,1],
		  [1,1,1,1,3,3,2],
		  [1,1,1,2,1,2,1],
		  [1,1,1,2,1,2,2],
		  [1,1,1,2,2,3,1],
		  [1,1,1,2,2,4,1],
		  [1,1,1,2,3,1,2],
		  [1,1,2,1,1,1,2],
		  [0,1,2,1,1,2,1],
		  [0,1,2,1,1,3,1],
		  [0,1,2,1,1,4,2],
		  [1,1,2,1,2,1,1],
		  [0,1,2,1,2,3,1],
		  [0,1,2,1,2,3,2],
		  [0,1,2,1,2,4,2] ]
	print ("\n".join( map (str, encode_int_matrix(m))))

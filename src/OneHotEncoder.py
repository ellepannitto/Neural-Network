#/usr/bin/python3
#

def encode_col (col, col_minimum, col_maximum):
	encoded_matrix = []
	for c in col:
		row = [0]*(col_maximum-col_minimum+1)
		row[c-col_minimum] = 1
		encoded_matrix.append(row)
	return encoded_matrix

def encode_int_matrix ( matrix ):
	new_matrix = [[] for _ in matrix]
	for c in range (len(matrix[0])):
		col = [row[c] for row in matrix]
		col_maximum = max( col )
		col_minimum = min( col )
		encoded_col = encode_col (col, col_minimum, col_maximum)
		for i,row in enumerate(encoded_col):
			new_matrix[i].extend(row)
	return new_matrix

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

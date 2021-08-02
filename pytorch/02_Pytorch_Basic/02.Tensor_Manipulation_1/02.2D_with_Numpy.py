import numpy as np

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]

print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)
# Rank  of t:  2
# Shape of t:  (4, 3)
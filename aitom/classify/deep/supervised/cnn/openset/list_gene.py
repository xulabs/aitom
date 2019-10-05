import os
import numpy as np

filename = './qiang_train.txt'
address = './qiang_train'

for i in os.listdir(address):
	temp = os.listdir( str(os.path.join(address, i)))
	for j in temp:
	    final = os.path.join(  str(os.path.join(address, i)), j)
	    with open(filename,'a') as fileobject:
	        fileobject.writelines(str(final)+'\n')
fileobject.close()

a = np.loadtxt(filename)
print(a[0:10])



import os
import numpy as np
import random

filename = './test_qiang_pair.txt'
# generate the paired images. 
'''
for item in os.listdir('./qiang_test'):
	data = list( os.listdir( str(os.path.join( './qiang_test', item  ))))
	print (len(data))
	for num in range(1000):

		pair = random.sample(data , 2)
		for i in range(len(pair)):
			pair[i] = pair[i][:-4]

		with open(filename,'a') as fileobject:
			fileobject.writelines(str(item)+' '+str(pair[0])+' '+str(pair[1])+'\n')
'''
profile = [3,4,5]

for item in os.listdir('./qiang_test'):
	data = list( os.listdir( str(os.path.join( './qiang_test', item  ))))
	print (len(data))
	for num in range(2000):
		pair = random.sample(data , 2)
		pair_2 = random.sample(profile , 2)
		for i in range(len(pair)):
			pair[i] = pair[i][:-4]

		with open(filename,'a') as fileobject:
			fileobject.writelines(str(pair_2[0])+' '+str(pair[0])+' '+str(pair_2[1])+' '+str(pair[1])+'\n')







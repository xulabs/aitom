import numpy as np 
import matplotlib.pyplot as plt

#load corr and orientation
c = np.load('./template_matching_tutorial/results/test_id-c.npy') #(10, 400, 398)
#phi = np.load('./template_matching_tutorial/results/test_id-phi.npy')
#psi = np.load('./template_matching_tutorial/results/test_id-psi.npy')
#theta = np.load('./template_matching_tutorial/results/test_id-theta.npy')

#view
for i in range(c.shape[0]):
	plt.subplot(2,5,i+1)
	plt.imshow(c[i])
	plt.title(i)
	plt.axis('off')
plt.tight_layout()
plt.show()

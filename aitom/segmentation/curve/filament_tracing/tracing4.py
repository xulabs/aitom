'''
Implemented based on:
Rigort, A., Günther, D., Hegerl, R., Baum, D., Weber, B., Prohaska, S., Medalia, O., Baumeister, W., & Hege, H. C. (2012). 
	Automated segmentation of electron tomograms for a quantitative description of actin filament networks. Journal of structural biology, 177(1), 135–144. 
	https://doi.org/10.1016/j.jsb.2011.08.012
Test data from:
Vinzenz, M., Nemethova, M., Schur, F., Mueller, J., Narita, A., Urban, E., Winkler, C., Schmeiser, C., Koestler, S. A., Rottner, K., Resch, G. P., Maeda, Y., & Small, J. V. (2012). 
	Actin branching in the initiation and maintenance of lamellipodia. Journal of cell science, 125(Pt 11), 2775–2785. https://doi.org/10.1242/jcs.107623
The search cone can be found at: link to sc.rec to be added
'''
import numpy as np
import aitom.geometry.rotate as GR
import aitom.geometry.ang_loc as AA
import math
import mrcfile
import sys

def angle_between(u, v): 
	'''
	@param: u: vector as a np.array
	@param: v: vector as a np.array
	@return: angle between u and v in radian
	'''
	cos = np.dot(u,v)/(np.linalg.norm(u)* np.linalg.norm(v))
	angle = np.arccos(np.clip(cos, -1, 1)) 
	return angle

def similarity(x, x0, corr, sigma_c, sigma_l, sigma_d):
	'''
	Similarity function, i.e. "the likelihood for a voxel x to be part of the same filament" as x0
	@param x: index of voxel x
	@param x0: index of voxel x0
	@param corr: correlation coefficient at voxel x from template matching 
	@param sigma_c: user-defined parameter for smoothness (smoother lines with smaller sigma_c)
	@param sigma_l: user-defined parameter for linearity
	@param sigma_d: user-defined parameter for distance

	@return: a similarity value that measures the likelihood for x0 and x to be part of the same filament
	'''
	#first get vector x-x0
	vec = np.array([x[0]-x0[0], x[1]-x0[1], x[2]-x0[2]]) #vec = x - x0 but as a np.array
	beta = angle_between(vec, orientation[x0])
	gamma = angle_between(vec, orientation[x])

	#Gaussian functions
	#co-circularity: smootheness
	C = math.exp((-(beta-gamma)**2)/(sigma_c**2))
	#linearity
	L = math.exp((-(beta+gamma)**2)/(sigma_l**2))
	#distance
	D = math.exp((-(np.linalg.norm(vec)**2))/(sigma_d**2))

	#similarity function
	s = corr * C * L * D
	return s

def convert(index):
	'''
	converts index in the form of a numpy array to a tuple
	e.g. (array([0]),array([0]),array([0]))-->(0,0,0)
	@param index: the index to be converted 

	@return: a index tuple
	'''
	return (index[0][0], index[1][0], index[2][0])

def forward(x0, sc): 
	'''
	The forward tracing algorithm
	@param x0: index of the starting voxel
	@param sc: the forward (along y) search cone as a numpy array

	@returns when search_result.max() <= t1, where t1 is a similarity threshold specified by the user

	updates global binary mask variable bm
	'''
	angles = (phi[x0], theta[x0], psi[x0]) #get ZYZ template rotation angles
	sc_curr = GR.rotate(sc, angle=angles, default_val= 0.0) #rotate search cone according to template rotation angles
	x0_loc = convert(np.where(sc_curr==sc_curr.min())) #get x0 location on the search cone for index calculation later
	sc_mask = sc_curr > 0 #convert the format of sc: >0-->True
	#create a np array the size of the sc to store search results
	#highest similarity in search_result means this voxel is part of the filament
	search_result = np.full(sc_mask.shape, -1.0) 

	#for every voxel in the search cone, calculate similarity and update search_result
	for idx, val in np.ndenumerate(sc_mask): 
		if val == True: 
			x = (x0[0]-x0_loc[0]+idx[0], x0[1]-x0_loc[1]+idx[1], x0[2]-x0_loc[2]+idx[2]) #x = idx - x0_loc + x0 #convert to the index of x
			if (x[0] <= c.shape[0]-1) and (x[1] <= c.shape[1]-1) and (x[2] <= c.shape[2]-1): #z, y, x boundaries
				s = similarity(x0, x, c[x], sigma_c, sigma_l, sigma_d) #calculate similarity between x and x0
				if s > t1:
					search_result[idx] = s #store voxel location when similarity is above threshold t1

	if search_result.max() > t1: #continue forward()
		x = convert(np.where(search_result==search_result.max())) #highest similarity in the search cone
		x = (x0[0]-x0_loc[0]+x[0], x0[1]-x0_loc[1]+x[1], x0[2]-x0_loc[2]+x[2]) #convert coordinate
		bm[x] = True #update binary mask
		return forward(x, sc) #recursively move forward

	else: #stop searching when all similarity values within sc are below threshold t1
		return

def backward(x0, sc): 
	'''
	The backward tracing algorithm
	@param x0: index of the starting voxel
	@param sc: the backward (along -y) search cone as a numpy array

	@returns when search_result.max() <= t1, where t1 is a similarity threshold specified by the user

	updates global binaey mask variable bm
	'''
	angles = (phi[x0], theta[x0], psi[x0])
	#rotate search cone according to template orientation
	sc_curr = GR.rotate(sc, angle=angles, default_val= 0.0) #search cone rotation
	x0_loc = convert(np.where(sc_curr==sc_curr.min())) #get x0 location on the search cone for index calculation later
	sc_mask = sc_curr > 0 #convert the format of sc: >0-->True
	#create a np array the size of the sc to store search results
	#highest similarity in search_result means this voxel is part of the filament
	search_result = np.full(sc_mask.shape, -1.0) 

	#for every voxel in the search cone, calculate similarity and update search_result
	for idx, val in np.ndenumerate(sc_mask): 
		if val == True: 
			x = (x0[0]-x0_loc[0]+idx[0], x0[1]-x0_loc[1]+idx[1], x0[2]-x0_loc[2]+idx[2]) #x = idx - x0_loc + x0 #convert to the index of x
			if (x[0] <= c.shape[0]-1) and (x[1] <= c.shape[1]-1) and (x[2] <= c.shape[2]-1): #z, y, x boundaries
				s = similarity(x0, x, c[x], sigma_c, sigma_l, sigma_d) 
				if s > t1:
					search_result[idx] = s #store voxel location when similarity is above threshold t1

	if search_result.max() > t1: #continue backward()
		x = convert(np.where(search_result==search_result.max())) #highest similarity in the search cone
		x = (x0[0]-x0_loc[0]+x[0], x0[1]-x0_loc[1]+x[1], x0[2]-x0_loc[2]+x[2]) #convert coordinate
		bm[x] = True #update binary mask
		return backward(x, sc) #recursively move backward

	else: #stop searching when all similarity values within sc are below threshold t1
		return

def trace():
	'''
	TODO: transfer testing to trace()
	'''
	pass

'''
#testing
sys.setrecursionlimit(10000) #forward() and backward() might reach recursion limit
t1 = 0.000001 #threshold for similarity
t2 = 0.0008 #threshold for correlation coefficient
sigma_c = 1.0 #smootheness
sigma_l = 1.0 #linearity
sigma_d = 1.0 #distance
sc_path = './y_cone.rec' #TODO: change filepaths
out_path = './bm.rec'
c_path = './results/test_id-c.npy'
phi_path = './results/test_id-phi.npy'
theta_path = './results/test_id-theta.npy'
psi_path = './results/test_id-psi.npy'

#load template matching results (see 007_template_matching_tutorial.py)
print("Loading template matching results...")
c = np.load(c_path) #(10, 400, 398)
phi = np.load(phi_path)
theta = np.load(theta_path)
psi = np.load(psi_path)
print("Map shape: ", c.shape)

#preprocess cross correlation matrix: negative values => 0.0
print("Preproces correlation coefficients...")
c [c<0.0] = 0.0
print("Maximum corr = ", c.max())

print("Preprocess orientations...")
orientation = np.empty(c.shape, dtype=object) #empty np array to store template orientations
v = np.array([0, 1, 0]) #assume template is parallel to the y axis
for idx, x in np.ndenumerate(c): #for all voxels
	angle = (phi[idx], theta[idx], psi[idx]) #get ZYZ angle for template orientation
	rm = AA.rotation_matrix_zyz(angle) #convert to rotation matrix
	orientation[idx] = rm.dot(v) #template orientation in [x, y, z]

bm = np.full(c.shape, False, dtype=bool) #binary mask to store tracing results

#read in search cone: SC is along the y axis, i.e. [0,1,0]
print("Prepare search cone...")
mrc = mrcfile.open(sc_path, mode='r+',permissive=True)
sc = mrc.data #rotated already, along the y axis for now #forward search cone
print('search cone size',sc.shape) #(10, 11, 11)
sc[(5,0,5)] = -100.0 #mark where x0 is with a negative value

rm = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]) #180 degree rotation of the search cone about the XYZ axes
sc_back = GR.rotate(sc, rm=rm, default_val= 0.0)#rotate to get the backward search cone

#tracing: go through corr for all c.max()>=t2
print("Start tracing actin filaments...")
while c.max() >= t2: 
	corr = c.max() #tracing starts with the highest corr
	x0 = np.where(c==corr) #get the index of c.max #e.g.(array([5]), array([177]), array([182]))
	assert (x0[0].size == 1), "multiple maximum"
	x0 = (x0[0][0], x0[1][0], x0[2][0]) #convert index to a tuple
	print("Now at ", x0, " with corr=", corr)
	bm[x0] = True

	forward(x0, sc) #recursively search forward
	backward(x0, sc_back)
	c[x0] = -1.0 # set c.max() to -1.0 and go to the next c.max()	

#output bm
bm = bm.astype(np.int16)
mrc = mrcfile.new(out_path, overwrite=True)
mrc.set_data(bm)
mrc.close()
print("DONE")
'''

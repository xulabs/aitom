# utility functions



'''
an alternative way to perform anistropic diffusion filtering is to use 
medpy.filter.smoothing.anisotropic_diffusion()
'''


import numpy as N
import warnings
import tomominer.filter.gaussian as FG


'''
perform standard anistropic diffusion filtering, modified according to filtering.anistropic_diffusion.fastaniso
'''
def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, gauss_smooth_sigma=None, step=(1.,1.,1.), option=1):
	"""
	3D Anisotropic diffusion.
	Usage:
	stackout = anisodiff(stack, niter, kappa, gamma, option)
	Arguments:
	        stack  - input stack
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (z,y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the middle z-plane will be plotted on every 
	        	 iteration
	Returns:
	        stackout   - diffused stack.
	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.
	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)
	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x,y and/or z axes
	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.
	"""

	# ...you could always diffuse each color channel independently if you
	# really want
	if stack.ndim == 4:
		warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
		stack = stack.mean(3)

	# initialize output array
	stack = stack.astype('float32')
	stackout = stack.copy()

	# initialize some internal variables
	deltaS = N.zeros_like(stackout)
	deltaE = deltaS.copy()
	deltaD = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	UD = deltaS.copy()
	gS = N.ones_like(stackout)
	gE = gS.copy()
	gD = gS.copy()


	for ii in range(niter):

		# calculate the diffs
		deltaD[:-1,: ,:  ] = N.diff(stackout,axis=0)
		deltaS[:  ,:-1,: ] = N.diff(stackout,axis=1)
		deltaE[:  ,: ,:-1] = N.diff(stackout,axis=2)

                if gauss_smooth_sigma is not None:
                    # a small modification: to smooth the diffs a bit. This sometimes is useful when the noise high
                    deltaD = FG.smooth(deltaD, sigma=gauss_smooth_sigma) 
                    deltaS = FG.smooth(deltaS, sigma=gauss_smooth_sigma) 
                    deltaE = FG.smooth(deltaE, sigma=gauss_smooth_sigma) 

		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gD = N.exp(-(deltaD/kappa)**2.)/step[0]
			gS = N.exp(-(deltaS/kappa)**2.)/step[1]
			gE = N.exp(-(deltaE/kappa)**2.)/step[2]
		elif option == 2:
			gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
			gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
			gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

		# update matrices
		D = gD*deltaD
		E = gE*deltaE
		S = gS*deltaS

		# subtract a copy that has been shifted 'Up/North/West' by one
		# pixel. don't as questions. just do it. trust me.
		UD[:] = D
		NS[:] = S
		EW[:] = E
		UD[1:,: ,: ] -= D[:-1,:  ,:  ]
		NS[: ,1:,: ] -= S[:  ,:-1,:  ]
		EW[: ,: ,1:] -= E[:  ,:  ,:-1]

		# update the image
		stackout += gamma*(UD+NS+EW)


	return stackout

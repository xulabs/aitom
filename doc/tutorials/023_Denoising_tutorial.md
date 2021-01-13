# Denoising Tutorial
In this tutorial, an example of denoisng volume data using AITom is shown in denoising.py. This tutorial uses no glucose ribosome tomograms provided by Freyberg Lab as examples. With appropriate parameter adjustments, this tutorial code can be used on pre-processing other cellular structure as well.

## Gaussian Denoising
The first type of denoising performed in this tutorial is Guassian denoising. There are three different options: *smooth*, *dog\_smooth* and *dog\_smooth\_large\_map*. *Smooth* is using python built-in gaussian filter function from *scipy.ndimage*. *Dog_smooth* is a difference of guassian filter with a factor of 1.1. *Dog\_smooth\_large\_map* convolves with a DoG function, delete unused data when necessary
    in order to save memory for large maps. The example results are shown in the result section.

## Bandpass Filter
The second type of denoisng performed in this tutorial uses a bandpass filter which is built with FFT functions. This denoising method was used in [*De novo visual proteomics in single cells through pattern mining*](https://arxiv.org/abs/1512.09347). The example results are shown in the result section.

## Anistropic Diffusion
The third type of denoisng performed in this tutorial is Anistropic Diffusion. The original code can be download from http://pastebin.com/sBsPX4Y7. The example results are shown in the result section.

## Input
denoising.py contains the following input values in main():  
- path - absolute path of the tomogram that needs to be denoised  
- name - name of the the tomogram which will be used for creating result name later
- G_type - 1: smooth; 2: dog\_smooth; 3: dog\_smooth\_large\_map
- gaussian_sigma: deviation value used in guassian denoising
- curve - tuple of factors will be applied to the volume data in the bandpass filter
- niter - number of iterations
- kappa - conduction coefficient between 20-100
- gamma - max value of .25 for stability
- step - tuple, the distance between adjacent pixels in (z,y,x)
- option - 1: Perona Malik diffusion equation No 1; 2: Perona Malik diffusion equation No 2
- ploton - if True, the middle z-plane will be plotted on every iteration
- result - absolute path of the output directory

## Tutorial Outline
1. Modify the input parameters marked by the \#TODO comments in denoising.py. For a description of the input parameters, see the Input section.
2. Run denoisng.py:  
python \<filepath\>/denoising.py  
Within denoising.py, steps are:
  * load the tomograms (INS\_HG\_21\_g1\_t1\_noglucose.rec)
  * denoise with picked type guassian filter
  * compute difference between original and guassian denoised tomograms (slice)
  * denoise with bandpass filter
  * compute difference between original and bandpass denoised tomograms (slice)
  * denoise with Anistropic Diffusion
  * compute difference between original and Anistropic Diffusion denoised tomograms (slice)

3. In the user-specified output directory, the output files are generated as:
  * original.png
  * \<tomogram name\>\_sig=\<value\>\_G\_type=\<value\>.png
  * \<tomogram name\>\_sig=\<value\>\_G\_type=\<value\>.rec
  * \<tomogram name\>\_BP.png
  * \<tomogram name\>\_BP.rec
  * \<tomogram name\>\_AD\_i=\<iterations\>\_k=\<kappa\>\_g=\<gamma\>.png
  * \<tomogram name\>\_AD\_i=\<iterations\>\_k=\<kappa\>\_g=\<gamma\>.rec

For further analysis, the .rec files can be read with the aitom.io.mrcfile_proxy.read_data(path) function. 

## Result
Here is the original image of 'INS_HG_21_g1_t1_noglucose'

!['INS_HG_21_g1_t1_noglucose_Original.png](https://user-images.githubusercontent.com/74321858/104505243-9a3a0c80-55b1-11eb-8f3a-f6f97000f130.png)

Here is the Gaussian denoising image with sigma of 2.5

!['INS_HG_21_g1_t1_noglucose_sig=2.5_G_type=1.png](https://user-images.githubusercontent.com/74321858/104506205-108b3e80-55b3-11eb-92ba-b46cea45faf5.png)

The difference between Gaussian denoising image and the original one is shown below. We can see that this removed some noise and emphasizes the boundary of the cellular structure significantly.

!['INS_HG_21_g1_t1_noglucose_gussain_difference.png](https://user-images.githubusercontent.com/74321858/104506583-98714880-55b3-11eb-94aa-30641254b5e0.png)

Here is the Anistropic Diffusion image with 20 iterations ,kappa=50,gamma=0.1, and using Perona Malik diffusion equation (option=1)

!['INS_HG_21_g1_t1_noglucose_AD_i=20_k=50_gamma=0.1.png](https://user-images.githubusercontent.com/74321858/104506924-074ea180-55b4-11eb-8fb8-550ed5b60289.png)

The difference between Anistropic Diffusion denoising image and the original one is shown below. We can see that this removed a significant amount of noise from the image while preserving the definition of the cellular structure.

!['INS_HG_21_g1_t1_noglucose_gussain_difference.png](https://user-images.githubusercontent.com/74321858/104507256-77f5be00-55b4-11eb-83d9-6504a69af4ec.png)


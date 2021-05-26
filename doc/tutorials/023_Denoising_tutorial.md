# Denoising Tutorial
In this tutorial, an example of denoisng volume data using AITom is shown in denoising.py. This tutorial uses [*aitom_demo_single_particle_tomogram*](https://cmu.app.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp/file/509290945451) as example. With appropriate parameter adjustments, this tutorial code can be used on pre-processing other cellular structure as well.

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
  * \<tomogram name\>\_sig=\<value\>\_G\_type=\<value\>.mrc
  * \<tomogram name\>\_BP.png
  * \<tomogram name\>\_BP.mrc
  * \<tomogram name\>\_AD\_i=\<iterations\>\_k=\<kappa\>\_g=\<gamma\>.png
  * \<tomogram name\>\_AD\_i=\<iterations\>\_k=\<kappa\>\_g=\<gamma\>.mrc

For further analysis, the .mrc files can be read with the aitom.io.mrcfile_proxy.read_data(path) function. 

## Result
Here is the original image of 'aitom_demo_single_particle_tomogram'

!['aitom_demo_single_particle_tomogram_Original.png](https://user-images.githubusercontent.com/74321858/106344888-77c71500-627a-11eb-8d90-f0f6fa04ef91.png)

Here is the Gaussian denoising image with sigma of 2.5

!['aitom_demo_single_particle_tomogram_G=2.5_type=1.png](https://user-images.githubusercontent.com/74321858/106344970-ee641280-627a-11eb-8fbf-602102fe2781.png)

The difference between Gaussian denoising image and the original one shows that this method removed some noise and emphasizes the boundary of the cellular structure significantly.


Here is the Bandpass denoising image 

!['aitom_demo_single_particle_tomogram_BP.png](https://user-images.githubusercontent.com/74321858/106344993-08055a00-627b-11eb-9565-f71859efc089.png)

The difference between Bandpass denoising image and the original one shows that the bandpass greatly heightens the contrast between the noise and the cells while blurs the boundary of cellular structure


Here is the Anistropic Diffusion image with 70 iterations ,kappa=100,gamma=0.25, and using Perona Malik diffusion equation (option=1)

!['aitom_demo_single_particle_tomogram_AD_i=70_k=100_g=0.25.png](https://user-images.githubusercontent.com/74321858/106344790-e788d000-6279-11eb-946a-dc5219485ea3.png)

We can see that this removed a slight amount of ambient noise from the image.


The demo single particle tomogram can be downloaded from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp.

!['particle_picking_result.png'](https://user-images.githubusercontent.com/17937329/71565767-ef2ae380-2a7f-11ea-9735-c9007a2553dc.png) 

'particle_picking_result.png' shown above is the result under such conditions:

1. 'sigma1' is set to 5 for better performance.

2. 'FG.dog_smooth' rather than 'FG.dog_smooth__large_map' is used in line 91 of https://github.com/xulabs/aitom/blob/master/aitom/pick/dog/particle_picking_dog__util.py. 
Using the latter leads to much more peaks.

3. 'pick_num' is set to 3000.

4. Selected peaks are marked on DoG-smoothed image.

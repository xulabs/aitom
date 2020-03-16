The demo subtomograms (aitom_demo_subtomograms.pickle) can be downloaded from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp.  

Given a tomogram/subtomogram, guess the tilt angle range and light axis.  
Input: 3-D array  
Output: tilt angle and light axis  
Note:  
In most cases, the tilt angle is around 60 degrees. 'tilt_angle_scan_range' is the scan range and should contain 60 degrees (e.g. [1,89]).  

For demo subtomogram ('5T2C_data'), the result is as follows:  
{'light_axis': 2, 'tilt_axis': 1, 'cor': 0.796683271884446, 'ang1': -65, 'ang2': 64}

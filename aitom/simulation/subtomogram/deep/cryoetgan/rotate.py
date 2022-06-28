# -*- coding=utf-8 -*-
from tomominer.geometry.rotate import rotate
import tomominer.geometry.ang_loc as AAL
import numpy as N
import cPickle as pickle
import os

def generate(v,num):
    vs_train = []
 
    for i in range(num):
        print '\r', i, '         ',
        angle = AAL.random_rotation_angle_zyz()
        loc_max = N.array(list(v.shape), dtype=float) * 0.2
        loc_r = (N.random.random(3)-0.5)*loc_max

        vs_train.append(rotate(v, angle=angle, loc_r=loc_r, default_val=v.mean()))
    return N.array(vs_train)


with open("situs_maps_40.pickle", "rb") as f:
    den = pickle.load(f)
with open("situs_maps_11.pickle","rb") as f:
    den2 = pickle.load(f)

#"31" "33", "35", "43", "69", "72","73"
count = 0
new_den = {}
#den_gen = generate(density_map)
for idx, img_den in den.items():
    print(count)
    #print(idx,len(img_den))
    count+=1
    if len(img_den) == 35:
        img_den = N.array(img_den)
        print(img_den.shape) 
        same_den = generate(img_den, 5)
        print(idx+": ", same_den.shape)
        new_den[idx] = same_den
for idx, img_den in den2.items():
    print(count)
    count+=1
    img_den = N.array(img_den)
    same_den = generate(img_den,5)
    new_den[idx] = same_den
print("done")
print(len(new_den))
#/shared/xiangruz/classification/experimental/Noble
with open("./density_map_40_5.pickle","w") as f:
    pickle.dump(new_den,f)


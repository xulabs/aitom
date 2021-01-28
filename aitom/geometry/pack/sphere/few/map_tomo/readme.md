#workflow

######run simu_subtomo.py, and a simulated subtomogram of a target macromolecule will obtained. 


pdb2map

```
pdb2map(op)
return ms: dict of density map
```


merge_map

```
random_rotate(v)
v: one density map
return vr, angle
```

```
angle_rotate(v, angle)
return vr
```

```
merge_map(v, protein_name, x, y, z, box_size)
v: list of density maps
protein_name: list of protein names
x,y,z: list of center coordinate
return hugemap,angle_list
```

```
trim_margin(hugemap)
return trimmed_cub
```

```
trim_target(hugemap, center, target_size = 30)
return targetmap

```

iomap

```
map2mrc(map, file)
dir: end with /
```

```
map2npy(map, file)
```

```
map2png(map, file)
```

```
readMrcMapDIR(dir)
dir: mrc maps in dir
return v: dic of maps {'nameXXX':map,...}
```

```
readMrcMap(file)
return mrc.data
```


map2tomogram

```
map2tomo(map, op)
return vb
```


mrc2singlepic

```
mrc2singlepic(mrcfile, pngdir, pngname='', view_dir=1)
pngdir: end with /
```
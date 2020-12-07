## Introduction
Three.js is a cross-browser JavaScript library and Application Programming Interface used to create and display animated 3D computer graphics in a web browser. (https://threejs.org/) We designed a [CryoEM render](https://github.com/xulabs/aitom/tree/master/aitom/pick/plot/particle_location_display__three_js) for displaying 3D CryoEM image.

![image](https://user-images.githubusercontent.com/25089434/73971181-6871ed80-48ec-11ea-8778-462f81f673ec.png)

## Use it in the following way
1. Download and save a MRC file. We provide a [test.mrc](https://github.com/xulabs/aitom_doc/issues/22) file. 
2. Replace the directory your local mrc file in MRC-render.html, line 53. 
```
# change the const variable to the directory of your mrc file
const url = 'models/test.mrc'; 
```
3. If you want to run locally, in order to circumvent same origin policy, following the [offical guidence](https://threejs.org/docs/#manual/en/introduction/How-to-run-things-locally) by three.js. 

## Reference
Thanks aearanky's [previous work](https://github.com/aearanky/MRC-file-renderer-v1).

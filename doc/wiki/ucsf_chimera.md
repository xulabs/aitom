Tips of using UCSF Chimera

Visualizing 3D volume (Tomograms in .rec/.mrc file):
  from Volume Viewer, select solid as Style.
  from Volume Viewer -> Features -> Planes, select One.
  then adjust the level and plane in Volume Viewer
  
  to enhance the contrast, can use the Gaussian filter in Volume Viewer -> Tools -> Volume Filter
 
 
  
Visualizing 3D subvolume iso-surface (Subtomograms in .rec/.mrc file):
  if the darker part (smaller values) corresponds to structural region,
    need to invert the intensity: open command line in favorites (upper bar), and type "vop scale #0 factor -1". (#0 denotes the index of the structure)
  
  to get rid of fractured structures likely to be noise, use Hide Dust in Volume Viewer -> Tools.
  
  modify the lighting condition: Favorites (upper bar) -> Side View -> Lighting
  
  modify the color in Volume Viewer color, can use hex color codes directly.
  
  
  
Visualizing multiple structures (multiple .rec/.mrc file):
  Use upper bar Favorites -> Model Panel to hide/dispaly and activate/deactivate a certein structure.
  Use upper bar Volume -> Fit in Map to align two structures. May need coarsely align the two target structures manually first.
  
  
  
Generate models from PDB files:
  for atomic model, use upper bar File -> Fetch by ID, and type the PDB ID. This may take several minutes.
  for generating iso-surface from atomic model:
    open command line in favorites (upper bar), and type "molmap #0 25". (#0 denotes the index of the structure, 25 denotes the desired resolution in Angstrom)



Snapshot current screen:
  File (upper bar) -> Save Image
  
  
  
Scale bar:
  Tools (upper bar) -> Higher-Order Structure -> Scale Bar
  
  
  
Background color:
  Tools (upper bar) -> Preferences -> Background (select in Category)
  


Save PDB file:
  Download pdb format-like files (tar.gz) from PDB if the macromolecule is of verg large structure. Open multiple pdb files in chimera and save a single one using  File (upper bar) -> Save PDB.

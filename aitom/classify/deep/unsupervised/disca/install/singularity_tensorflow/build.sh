#!/bin/bash

echo "Building Singularity image DISCA.sif..."
singularity build --fakeroot --force DISCA.sif DISCA.def

echo -e "\nBuild complete. Run 'singularity exec DISCA.sif <command>' to execute a command inside the container.\n"
echo -e "\nTo interactively enter the container with mounted directory, Run 'singularity shell --nv -B /location/to/directory:/mnt ./DISCA.sif\n'
echo -e "You may need to modify the bind flag (-B) to make a directory outside of /home available to the container.\n"
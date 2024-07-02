#The following Sigularity Container uses singularity-ce-3.10.3

To build Tensorflow DISCA container Copy DISCA.def and build.sh to a local directory

Run the following command in the same directory:

./build.sh 

The output of build.sh is DISCA.sif

Run 'singularity exec DISCA.sif <command>' to execute a command inside the container

You may need to modify the bind flag (-B) to make a directory outside of /home available to the container

To interactively enter the container with mounted directory, Run 'singularity shell --nv -B /location/to/directory:/mnt ./DISCA.sif


#If Sigularity is not available in your system, please perform the following prior to the installation (you will need sudo privileges):

1) 
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup \
    libglib2.0-dev

2) 
wget https://github.com/sylabs/singularity/releases/download/v3.10.3/singularity-ce-3.10.3.tar.gz

3)
tar -xzf singularity-ce-3.10.3.tar.gz && \
    cd singularity

4)
#please go to https://go.dev/dl/ to select a suitable OS and Architecture, for the case of Linux OS and x86-64 Arc you can run the following: 
wget https://dl.google.com/go/go1.22.4.linux-amd64.tar.gz && \
  sudo tar -C /usr/local -xzvf go1.22.4.linux-amd64.tar.gz && \ 
  rm go1.22.4.linux-amd64.tar.gz

5)
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc

6)
. ~/.bashrc

7)
 ./mconfig && \
    make -C builddir && \
    sudo make -C builddir install





## Pre-requisites
1. Conda installed
   ```
   # test conda
   conda --version
   ```

## Installation
1. Install Scipion (https://scipion-em.github.io/docs/docs/scipion-modes/install-from-sources#)

2. Install scipion-em-xmipp (https://scipion-em.github.io/docs/docs/scipion-modes/install-from-sources#step-4-installing-xmipp3-and-other-em-plugins)

3. Install scipion-em-tomo (in developement):
   ```
   git clone https://github.com/scipion-em/scipion-em-tomo
   scipion installp -p local/path/to/scipion-em-tomo --devel
   # Note that 'local/path/to/scipion-em-tomo' indicates that you should pick a local path of scipion-em-tomo.
   ```

4. Install AITom plugin in devel mode:

   Note:
   1. Do not have conda in the path (remove any conda initiatization in the bash.rc or equivalent).
   2. Tell Scipion how to activate conda by setting the variable CONDA_ACTIVATION_CMD to something like: I have this working for my laptop in my <SCIPION_HOME>/config/scipion.conf , under [BUILD] section : CONDA_ACTIVATION_CMD= eval "$(/extra/miniconda3/bin/conda shell.bash hook)"
   3. The plugin will create a conda VE('aitom-0.0.1') and install AITom automatically.
   ```
   # install AITom plugin
   git clone https://github.com/scipion-em/scipion-em-aitom
   scipion installp -p local/path/to/scipion-em-aitom --devel
   
   # unistall
   # scipion uninstallp -p scipion-em-aitom
   ```

5. Test the plugin:
   ```
   scipion test aitom.tests.tests_picking.TestAitomPicking
   ```
   

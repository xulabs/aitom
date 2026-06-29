# TomoPicker: PU Learning-based Macromolecule Picking in Cryo-Electron Tomograms

**Localization of macromolecules in crowded cellular cryo-electron tomograms from extremely sparse labels**  
*Mostofa Rafid Uddin, Ajmain Yasar Ahmed, H M Shadman Tabib, Md Toki Tahmid, Md Zarif Ul Alam, Zachary Freyberg, Min Xu*  
Published in *Briefings in Bioinformatics*, Volume 26, Issue 6, November 2025  
[Paper](https://academic.oup.com/bib/article/26/6/bbaf630/8351049)

## Model and Dataset Downloads

Before getting started, download the following resources:

- **Model Checkpoint**: [pombe_vpp_ribosome_unet_GE_KL.pt](https://zenodo.org/records/18187955/files/pombe_vpp_ribosome_unet_GE_KL.pt?download=1)
- **Dataset (Tomogram)**: [TS_0003.mrc](https://drive.google.com/file/d/1_EYWWb5PFBNXyJRSiWPabegV1UJTRcUj/view?usp=sharing)
- **Inference Results (Example)**: [Download inference results](https://drive.google.com/file/d/16N70Skjhhec00YO0Dkdnc8osY63ooukH/view?usp=drive_link)

Place the downloaded model checkpoint in a `Model/` folder and the tomogram in a `Data/` folder (these folders will be created automatically if they don't exist).

## Installation


To install the environment:

```sh
conda create -n tomopicker -c conda-forge python=3.8.3 -y
conda activate tomopicker
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
pip install -r requirement.txt
```

## Data Preparation

We recommend using tools like Warp to reconstruct and denoise tomograms before running TomoPicker. Ensure that your tomograms are properly preprocessed and in `.mrc` format.

## Getting Started

### Inference

After downloading the model checkpoint and preparing your tomogram, you can run inference in just a few lines:

```sh
python inference_tomopicker.py \
  --tomograms_path Data \
  --tomogram_name TS_0003.mrc \
  --encoder unet \
  --name pombe_vpp_ribosome_unet_GE_KL \
  --model_path Model \
  --pick 2000 \
  --nms \
  --radius 12 \
  --size 24
```

This command performs inference using a UNet-based model trained with the GE-KL objective. It will pick up particles from the tomogram `TS_0003.mrc` located in the `Data/` folder using the model `pombe_vpp_ribosome_unet_GE_KL.pt` stored in the `Model/` directory. This model is trained on 10 annotated macromolecules from the TS_0001 tomogram of *S. Pombe* cells in the VPP dataset.

**Output**:  
The predicted particle coordinates along with their scores will be saved in:

```
estimations/pombe_vpp_ribosome_unet_GE_KL/estimations_TS_0003.mrc_2000.csv
```

### Training

For training the model, you need to run `train_tomopicker.py` using python. You may run this Python script to train a PU learning-based model for picking macromolecular particles from tomograms.

#### Examples

- To see details on command-line arguments:
```sh
python train_tomopicker.py -h
# or
python train_tomopicker.py --help
```

- To run the script for creating a new input dataset and then training a model:
```sh
python train_tomopicker.py \
  --tomogram <tomogram_path> \
  --coord <annotated_coordinate_path> \
  --name <model_name> \
  --save_weight <path_for_saved_model> \
  --save_log <path_for_saved_train_log> \
  --objective <objective_type> \
  --encoder <Network_type> \
  --pick <expected_number_of_particles_in_the_training_tomogram> \
  --radius <particle_radius> \
  --size <subtomogram_size> \
  --make
```

- To run the script for training a model:
```sh
python train_tomopicker.py \
  --input <input_data_path> \
  --name <model_name> \
  --save_weight <path_for_saved_model> \
  --save_stat <path_for_saved_result> \
  --objective <objective_type> \
  --encoder <Network_type> \
  --pick <expected_number_of_particles_in_the_training_tomogram> \
  --radius <particle_radius> \
  --size <subtomogram_size>
```

**Note**: Three objective types are supported: PN, PU, and GE-KL. Two network types are supported: basic and unet. However, unet always gives superior results. In our experiments, we use only one tomogram for training and one for validation.

## Citing TomoPicker

If you find this project helpful for your research, please consider citing the following BibTeX entry:

```bibtex
@article{uddin2025localization,
  title={Localization of macromolecules in crowded cellular cryo-electron tomograms from extremely sparse labels},
  author={Uddin, Mostofa Rafid and Ahmed, Ajmain Yasar and Tabib, H M Shadman and Tahmid, Md Toki and Alam, Md Zarif Ul and Freyberg, Zachary and Xu, Min},
  note={*These authors contributed equally to this work.},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={6},
  pages={bbaf630},
  year={2025},
  publisher={Oxford University Press}
}
```

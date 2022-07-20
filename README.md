# NeILF: Neural Incident Light Field for Material and Lighting Estimation

This software project accompanies the research paper, [NeILF: Neural Incident Light Field for Physically-based Material Estimation](https://arxiv.org/abs/2203.07182v2) (ECCV 2022) and can be used to reproduce the results in the paper.

<p align="center">
  <img width="90%" src="docs/teaser.png"/>
</p>

## Introduction 
[NeILF](https://arxiv.org/abs/2203.07182v2) is a differentiable rendering framework for material and lighting estimation from multi-view images and a known geometry. If you find this project useful for your research, please cite:
```
@inproceedings{yao2022neilf,
    title={NeILF: Neural Incident Light Field for Physically-based Material Estimation},
    author={Yao Yao and Jingyang Zhang and Jingbo Liu and Yihang Qu and Tian Fang and David McKinnon and Yanghai Tsin and Long Quan},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```

## Getting Started

### Environment Setup

Install cuda and conda then:
```
conda create -n neilf python=3.8
conda activate neilf
conda install pytorch torchvision -c pytorch
cd ml-neilf/code
pip install -r requirement.txt
```
### Download
Download the Synthetic dataset, the preprocessed DTU dataset, and the preprocessed BlendedMVS dataset from [Box](https://apple.box.com/s/epkd7hamlpd7ltrsy4fyilzci19rz84r). Note that for the Synthetic dataset, all scenes shares the same geometry files and the BRDF ground truths. We only provide the geometry files and the BRDF GT in `synthetic_city`, and users may structure the input of another scene by substituting the image folder (`synthetic_city/inputs/images`) to another input image folder.

### Optimization
Download the datasets used in the paper (synthetic, DTU, and BlendedMVS). Choose one scene and run the following:
```
python training/train.py DATA_FOLDER OUTPUT_FOLDER --config_path CONFIG_PATH
```
Where `DATA_FOLDER` is the strutured input data folder, `OUTPUT_FOLDER` is the output folder, and `CONFIG_PATH` is the path to the config file of all parameters. 

We provide seperate configs for synthetic/BlendedMVS/DTU datasets (see provided `config/` in the codebase). If you encounter the out-of-memory issue, please decrease the `num_pixel_samples` (or the `num_train_incident_samples` or the `dims` in the two networks) in the config file.

### BRDF Export
Make sure you have correct texture atlas input before adding `--export_brdf`. Results will be stored in `OUTPUT_FOLDER/evaluation`:
```
python evaluation/evaluate.py DATA_FOLDER OUTPUT_FOLDER --config_path CONFIG_PATH --export_brdf
```

You can also pass `--eval_nvs` for novel view rendering evaluation and `--eval_brdf` for BRDF quality evaluation (if GT BRDF is provided). The PSNR and SSIM scores will be recorded and stored in `OUTPUT_FOLDER/evaluation/report_evaluation.json`.

### Batch Run
We provide a script to batch estimate all synthetic/BlendedMVS/DTU scenes.
```
bash sh/batch_run.sh synthetic 0
bash sh/batch_run.sh BlendedMVS 1
bash sh/batch_run.sh DTU 2
```
Please change `DATA_ROOT` and `OUTPUT_ROOT` in the script accordingly. The three datasets will be run at gpu0, gpu1, and gpu2 respectively (suppose your machine contains > 3 GPUs).


## File Formats

Input data should be strutured into the following:
```
.                          
├── inputs      
│   ├── images
│   ├── position_maps
│   ├── depth_maps (optional)
│   ├── normal_maps
│   ├── model (optional)
│   └── sfm_scene.json   
│      
└── ground_truths (optional)
    └── materials 
        ├── kd
        ├── roughness
        └── metallic       
```
### Camera Input
The `sfm_scene.json` file is used to stored the metadata and the SfM result of a scene. Please refer to `load_cams_from_sfmscene` in `utilts/io.py` for details.
* Camera intrinsics are stored in `sfm_scene['camera_track_map']['images']['INDEX']['intrinsic']`
* Camera extrinsics are stored in `sfm_scene['camera_track_map']['images']['INDEX']['extrinsic']`
* Image list are stored in `sfm_scene['image_list']['file_paths']`. (image index -> image path)
* The bounding box transformation is stored in `sfm_scene['bbox']['transform']`, which can transform the eight bounding box corners to normalized points at `{1/-1, 1/-1, 1/-1}`. It is used to compute the scale mat for coordinate normalization.
* Other fields could be ignored.

### Image Input
The image names should be stored in `sfm_scene['image_list']['file_paths']`. You can use `.jpg` and `.png` formats for LDR inputs, or `.exr` and `.tiff` formats for HDR inputs. 

### Geometry Input
The geometry input should be provided as either the rendered position map at each view (in `inputs/position_maps`) or the rendered depth map at each view (in `inputs/depth_maps`). 

### Texture Input (for BRDF Export)
The position atlas should be provided for exporting BRDF texture maps. (in `inputs/model`, see sample data for details)

## License
This sample code is released under the [LICENSE](LICENSE) terms. The Synthetic dataset is released under the [DATASET_LICENSE](DATASET_LICENSE)

## Changes to Paper Settings
For better reconstruction quality, the implementation of this repo is slightly different with the settings described in the original paper:
* Pixel samples in each training iteration is decreased from 16000 to 8192.
* Total Training iterations is increased from 15000 to 30000. Also, the iterations for learning rate decay are changed from [5000, 10000] to [10000, 200000]. 
* The weighting for Lambertian assumption loss is change from 1e-4 to 5x1e-3 for the Synthetic dataset, and 5x1e-5 for DTU and BlendedMVS dataset.
* The weighting for Smoothness loss is changed from 1e-3 to 5x1e-4
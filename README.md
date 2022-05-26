# Rethinking training of 3D GANs

[[Website]](https://rethinking-3d-gans.github.io)
[[Paper]](https://rethinking-3d-gans.github.io/rethinking-3d-gans.pdf)

![teaser](https://user-images.githubusercontent.com/105873229/170562989-bd409c04-bc49-4439-9b2e-7f6986eaa9e5.jpg)


## Installation
To install and activate the environment, run the following command:
```
conda env create -f environment.yml -p env
conda activate ./env
```
This repo is built on top of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), so make sure that it runs on your system.

## Data format
Data should be stored in zip archives, the exact structure is not important, the script will be used all the found images.
For FFHQ and Cats, we use the same data processing as [GRAM](https://yudeng.github.io/GRAM/).
Put your datasets into `data/` directory.

If you want to train with camera angles enabled, then create a `dataset.json` with `camera_angles` dict of `"<FILE_NAME>": [yaw, pitch, roll]` key/values.
Also, use `model.discriminator.camera_cond=true model.discriminator.camera_cond_drop_p=0.5` command line arguments (or simply override them in the config).
If you want to train on a custom dataset, then create its config under `configs/dataset` folder.

Data links:
- [Megascans Plants 256x256](https://www.dropbox.com/s/078gy1govyyoye9/plants_256.zip?dl=0)
- [Megascans Food 256x256](https://www.dropbox.com/s/lekkx0agd4fjaaa/food_256.zip?dl=0)

## Training

To launch training, run:
```
python src/infra/launch.py hydra.run.dir=. exp_suffix=min0.125-anneal10k-gamma0.05-dblocks3-cameracond-drop0.5-cin dataset=ffhq_posed dataset.resolution=512 training=patch_beta training.patch.min_scale_trg=0.25 training.patch.anneal_kimg=10000 model=eg3d training.metrics=fid2k_full env=local training.patch.resolution=64 model.discriminator.num_additional_start_blocks=3 training.kimg=100000 training.gamma=0.05 model.generator.tri_plane.res=512
```

To continue training, launch:
```
python src/infra/launch.py hydra.run.dir=. experiment_dir=<PATH_TO_EXPERIMENT> training.resume=latest
```

## Evalution
At train time, we compute FID only on 2,048 fake images (vs all real images), since generating 50,000 images takes too long.
To compute FID for 50k fake images, run:
```
python src/scripts/calc_metrics.py hydra.run.dir=. ckpt.networks_dir=<CKPT_DIR> script.data=<PATH_TO_DATA> script.mirror=true script.gpus=4 script.metrics=fid50k_full img_resolution=256 ckpt.selection_metric=fid2k_full ckpt.reload_code=true script=calc_metrics
```

## Visualization

We provide a lot of visualizations types, with the entry point being `configs/scripts/inference.yaml'.
For example, to create the main front grid, run:
```
python src/scripts/inference.py hydra.run.dir=. ckpt.networks_dir=<CKPT_DIR> vis=front_grid camera=points output_dir=<OUTPUT_DIR> num_seeds=16 truncation_psi=0.7
```

To create visualization videos, run:
```
python src/scripts/inference.py hydra.run.dir=. ckpt.networks_dir=<CKPT_DIR> output_dir=<OUTPUT_DIR> vis=video camera=front_circle camera.num_frames=64 vis.fps=30 num_seeds=16
```

## Rendering Megascans

To render the megascans, obtain the necessary models from the [website](https://quixel.com/megascans/home), convert them into GLTF, create a `enb.blend` Blender environment, and then run:
```
blender --python render_dataset.py env.blend --background
```
The rendering config is located in `render_dataset.py`.

# Music Style Transfer with Time-Varying Inversion of Diffusion Models

## Description
This repo contains code, data samples and user guide for our AAAI2024 paper ["Music Style Transfer with Time-Varying Inversion of Diffusion Models"](https://arxiv.org/abs/2402.13763). 

## Setup

Our code builds on, and shares requirements with [Textual Inversion (LDM)](https://github.com/rinongal/textual_inversion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate ldm
```

You will also need the official Riffusion text-to-image checkpoint, available through the [Riffusion project page](https://github.com/riffusion/riffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/sd/
wget -O models/ldm/sd/model.ckpt https://huggingface.co/riffusion/riffusion-model-v1/resolve/main/riffusion-model-v1.ckpt
```

## Usage

### Inversion

To invert an image set, run:

```
python main.py --base configs/stable-diffusion/v1-finetune.yaml
               -t 
               --actual_resume /path/to/pretrained/model.ckpt 
               -n <run_name> 
               --gpus 0, 
               --data_root /path/to/directory/with/style mel-spectrograms

```

In the paper, we use 3k training iterations. However, some concepts (particularly styles) can converge much faster.

Embeddings and output images will be saved in the log directory.

See `configs/stable-diffusion/v1-finetune.yaml` for more options, such as: changing the placeholder string which denotes the concept (defaults to "*"), changing the maximal number of training iterations, changing how often checkpoints are saved and more.


### Generation

To generate new images of the learned concept, run:
```
python scripts/txt2img.py --ddim_eta 0.0 
                          --n_samples 1 
                          --n_iter 2 
                          --scale 5.0 
                          --ddim_steps 50 
                          --strength 0.7
                          --content_path /path/to/directory/with/content mel-spectrograms
                          --embedding_path /path/to/logs/trained_model/checkpoints/ 
                          --ckpt_path /path/to/pretrained/model.ckpt 
                          --prompt "*"
```

where * is the placeholder string used during inversion.
### Performing conversion between mel-spectrograms and audio
Please refer to [Riffusion project page](https://github.com/riffusion/riffusion).
###  Data
We provide some samples of our data in ./images folder.

## Tips and Tricks
- Results can be seed sensititve. If you're unsatisfied with the model, try re-inverting with a new seed (by adding `--seed <#>` to the prompt).


## Results
Samples are available at [MusicTI](https://lsfhuihuiff.github.io/MusicTI/).

## Evaluation
We utilize [CLAP] (https://github.com/LAION-AI/CLAP) ( Contrastive Language-Audio Pretraining) for quantitative evaluation.

## Comparision
We compare the results obtained by running the official code and models provided for the following two methods on our collected dataset.
[SSVQVAE] (https://github.com/cifkao/ss-vq-vae)
[MUSICGEN] (https://github.com/facebookresearch/audiocraft)

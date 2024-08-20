# Paired Hierarchical Variational Autoencoders

This is an archive for the models used in our (unpublished) paper *Paired Hierarchical Variational Autoencoders for Image Translation and Cross Modality Conditional Synthesis*. We will include steps for reproducing the reported results.

As the model consists of paired HVAEs, the source code contains some sections which are heavily modified from [VDVAE](https://github.com/openai/vdvae). We also include some of the suggested changes from [Efficient-VDVAE](https://github.com/Rayhane-mamah/Efficient-VDVAE).

## Setup

We use `Python 3.10.12`, however, more recent versions should work fine. We have included a `requirements.txt` for setup of the appropriate virtual-environment. Below is an example to setup for a Linux system:

```
python3 -m venv venv
. venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data Preparation

We test the model on two datasets. The first dataset contains NDVI and RGB satellite imagery captured by Sentinel 2, while the second contains MRI/CT Scans from male pelvic regions.

### Dataset 1 - RGB/NDVI

Use of Copernicus Sentinel data allows for distribution, so we have made the processed dataset available [here](https://github.com/SigmaRichards/ndvi-rgb-i2i-data). Please see the [legal notice](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice).

By default, we make the data available in a directory labelled `datasets/SENT`. This can be changed, however, you will need to ensure to update the relevant locations in the script. No pre-processing of this data is required for reproducing our results.

### Dataset 2 - CT/MRI

This dataset has been made publicly available for academic use (see [here](https://doi.org/10.1002/mp.12748)). We are not permitted to redistribute the data, however, acquisition and processing details for this dataset are available [here](https://github.com/danhessres/phvae-mrict-preprocess). A request for access will need to be made [here](https://doi.org/10.5281/zenodo.583096).

Similarly as above, we make the data available in a directory labelled `datasets/GOLD`.

With both datasets prepared, the directory tree for datasets should be:
```
.
└── datasets
    ├── GOLD
    │   ├── test
    │   ├── train
    │   └── val
    └── SENT
        ├── test
        ├── train
        └── val
```

## Training

Note that these models are very computationally expensive to train, as the require large amounts of memory to optimize. Once these models are trained, they can be run on much more modest machines.

While the code as-is *should* run on CPU-only systems, it is not recommended. Additionally, the code has only been used on single-GPU systems, so multi-GPU systems aren't directly supported.

The code snippets below are sufficient for reproducing the results from the paper, which can be easily modified to suit your need. We have provided "argument-files" in `model_defs` to make model definitions easier. All the models below train for 100 epochs. By default, the models checkpoints will be placed in `artifacts`

### Training RGB/NDVI model
NLL (DMOL):
```
python3 train.py --argfile model_defs/ndvi-rgb-argfile.json --model_name ndvi-rgb-nll01 --data_dir datasets/SENT
```
MAE:
```
python3 train.py --argfile model_defs/ndvi-rgb-mae-argfile.json --model_name ndvi-rgb-mae01 --data_dir datasets/SENT
```

### Training CT/MRI model
NLL (DMOL):
```
python3 train.py --argfile model_defs/mri-ct-argfile.json --model_name mri-ct-nll01 --data_dir datasets/GOLD
```
MAE (*note: this model additionally trains for 1 warmup epoch otherwise it behaves poorly*):
```
python3 train.py --argfile model_defs/mri-ct-mae-argfile.json --model_name mri-ct-mae01 --data_dir datasets/GOLD
```

## Testing and Evaluation

The test script will start by running inference on the full test set. The outputs will be placed in a subdirectory of `results`. By default, it tests variants for both "best" and "latest" checkpoints of the model, with both temperature 1.0 and 0.0. All the performance results in the paper (unless explicitly stated) use "best" and "0.0". Inside the model variant directory (i.e., `results/[MODEL_NAME]/[VARIANT]/`) there a various subdirectories which contain image outputs:

 - *gtA*: ground-truth images for modality "A"
 - *gtB*: ground-truth images for modality "B"
 - *A2A*: reconstruction predictions for modality A
 - *B2B*: reconstruction predictions for modality B
 - *A2B*: translation predictions from modality A to modality B
 - *B2A*: translation predictions from modality B to modality A

After running inference, the script will run evaluation. It will calculate the pairwise PSNR and SSIM for each task (i.e., A2A, B2B, A2B, B2A) as well as the mean and standard deviation, and finally task-global FID. For FID calculation we use (a modified) [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) to avoid requiring Tensorflow as well. These values are output to a JSON file inside the variant directory (i.e., `results/[MODEL_NAME]/[VARIANT]/eval.log`).

### Testing RGB/NDVI model
NLL (DMOL):
```
python3 test.py --argfile model_defs/ndvi-rgb-argfile.json --model_name ndvi-rgb-nll01 --data_dir datasets/SENT
```
MAE:
```
python3 test.py --argfile model_defs/ndvi-rgb-mae-argfile.json --model_name ndvi-rgb-mae01 --data_dir datasets/SENT
```

### Testing CT/MRI model
NLL (DMOL):
```
python3 test.py --argfile model_defs/mri-ct-argfile.json --model_name mri-ct-nll01 --data_dir datasets/GOLD
```
MAE: 
```
python3 test.py --argfile model_defs/mri-ct-mae-argfile.json --model_name mri-ct-mae01 --data_dir datasets/GOLD
```

### Results

The following JSON files should contain all performance metrics.

RGB/NDVI:
 - NLL: `results/ndvi-rgb-nll01/best_t0.0/eval.log`
 - MAE: `results/ndvi-rgb-mae01/best_t0.0/eval.log`

CT/MRI:
 - NLL: `results/mri-ct-nll01/best_t0.0/eval.log`
 - MAE: `results/mri-ct-mae01/best_t0.0/eval.log`

## Result deviations

We have done everything we can to minimize the sources of deviations in the results where possible. State (and the parameter `random_state`) plays a large role in this, which is why the argfiles provided have a set state. However, there are other factors which may still cause deviations, some of which can be accounted for. Our testing involved 2 machines, one with a 3090, and the other with a 4090. Given the same model and `random_state`, our machines had identical testing/inference results. However, training a model on each machine separately resulted in slightly different results.

The first step that can be taken is forcing torch to use deterministic processes. In our testing, subsequent runs on the same machine resulted in *no change* in terms of training, and testing. The deviation we note between models trained on each machine separately was also not impacted. This can be achieved by placing the following in `setup.py` (as this is called by both `train.py` and `test.py`):

```
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

Another potential source of deviations can be due to `num_workers > 0`, however, we also found no impact on the deviation between our machines.

### Temperature Sampling and DMOL

As the testing script ties both LV sampling and DMOL sampling to the same temperature term, setting `temperature = 0` should result in deterministic inference/testing independent of whether `random_state` is set or not. However, the sampling for DMOL involves 2 sampling steps:

 1. A logistic mixture is sampled (e.g. from `dmol_mixtures` number of mixtures)
 2. A value is sampled from the logistic mixture

While both of these *can* be controlled by a temperature term, we made the decision to only directly use temperature in the second step. Thus, step 1 is still psuedo-random at `temperature = 0` for unknown `random_state`. This behaviour can be modified, however, it is not tested by us. The models we provide here strictly use `dmol_mixtures = 1`, thus the process should still be deterministic under these conditions.

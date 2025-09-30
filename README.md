# VSALip

Visual Speaker Authentication via Lip Motions: Appearance Consistency and Semantic Disentanglement

## Content
[Deep Lipreading](#deep-lipreading)
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [How to install the environment](#how-to-install-environment)
- [How to prepare the dataset](#how-to-prepare-dataset)
- [How to train](#how-to-train)
- [How to test](#how-to-test)
- [How to extract embeddings](#how-to-extract-embeddings)

[Model Zoo](#model-zoo)

[Citation](#citation)

[License](#license)

[Contact](#contact)


## VSALip Project
### Introduction
Lip-based visual biometric technology shows significant potential for improving the security of identity authentication in human-computer interaction. However, variations in lip contours and the entanglement of dynamic and semantic features limit its performance. To tackle these challenges, we revisit the personalized characteristics in lip-motion signals and propose a lip-based authentication framework based on personalized feature modeling. Specifically, the framework adopts a “shallow 3D CNN + deep 2D CNN” architecture to extract dynamic lip appearance features during speech, and introduces an appearance consistency loss to capture spatially invariant features across frames. For dynamic features, a semantic decoupling strategy is proposed to force the model to learn lip motion patterns that are independent of semantic content. Additionally, we design a dynamic password authentication method based on visual speech recognition (VSR) to enhance system security. In our approach, appearance and motion patterns are used for speaker verification, while VSR results are used for passphrase verification — they working jointly. Experiments on the ICSLR and GRID datasets show that our method achieves excellent performance in terms of authentication accuracy and robustness, highlighting its potential in secure human-computer interaction scenarios. 

<img src="README.assets/Fig.5.png" alt="图片描述" width="518">

### How to install the environment

This guide explains how to set up the environment for the LDWLip project, including Python environment, PyTorch framework, required libraries, and face detection/alignment models. Follow the steps sequentially.

1. **Download the project**  
   The project has not been uploaded to GitHub. Please contact the author to obtain `LDWLip.tar.gz (414.40 MB)`.

2. **Create and activate Conda environment**
```bash
conda create -y -n LDWLip python=3.9
conda activate LDWLip
````

3. **Install PyTorch**
   CPU version:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

GPU version (CUDA 11.7):

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

4. **Install required libraries**

```bash
pip install sentencepiece av six opencv-python scikit-image tqdm thop tensorboard pyyaml tiktoken chardet
pip install einops tensorboardX
```

> Note: The latest version of pip may conflict with fairseq. If installation fails, downgrade pip:

```bash
python -m pip install pip==24.0
```

5. **Install MMCV**

```bash
python -m pip install openmim
python -m mim install mmcv-full
```

6. **Install preprocessing models (optional, not required for training/evaluation)**
   Face detection:

```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull  # optional
pip install -e .
```

Face alignment:

```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull  # optional
pip install -e .
```

After completing these steps, the LDWLip environment is ready for data preprocessing, training, and evaluation.

```


### How to prepare the dataset




### How to train

1. Train a visual-only model.

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --config-path <MODEL-JSON-PATH> \
                                      --annonation-direc <ANNONATION-DIRECTORY> \
                                      --data-dir <MOUTH-ROIS-DIRECTORY>
```

2. Train an audio-only model.

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality audio \
                                      --config-path <MODEL-JSON-PATH> \
                                      --annonation-direc <ANNONATION-DIRECTORY> \
                                      --data-dir <AUDIO-WAVEFORMS-DIRECTORY>
```

We call the original LRW directory that includes timestamps (.txt) as *`<ANNONATION-DIRECTORY>`*.

3. Resume from last checkpoint.

You can pass the checkpoint path (`.pth` or `.pth.tar`) *`<CHECKPOINT-PATH>`* to the variable argument *`--model-path`*, and specify the *`--init-epoch`* to 1 to resume training.


### How to test

You need to specify *`<ANNONATION-DIRECTORY>`* if you use a model with utilising word boundaries indicators.

1. Evaluate the visual-only performance (lipreading).

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <MOUTH-ROIS-DIRECTORY> \
                                      --test
```

2. Evaluate the audio-only performance.

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --modality audio \
                                      --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <AUDIO-WAVEFORMS-DIRECTORY>
                                      --test
```



## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
need to update...
```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.


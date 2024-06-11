<h1 align="center"> Visual Representation Learning with Stochastic Frame Prediction</h1>
<div align="center">
  <a href="https://huiwon-jang.github.io/" target="_blank">Huiwon&nbsp;Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Dongyoung&nbsp;Kim</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://junsu-kim97.github.io/" target="_blank">Junsu&nbsp;Kim</a><sup>1</sup>
  <br>
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://people.eecs.berkeley.edu/~pabbeel/" target="_blank">Pieter&nbsp;Abbeel</a><sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://younggyo.me/" target="_blank">Younggyo&nbsp;Seo</a><sup>1,3</sup>
  <br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>UC Berkeley &emsp; <sup>3</sup>Dyson Robot Learning Lab &emsp; <br>
</div>
<h3 align="center">[<a href="https://sites.google.com/view/2024rsp">project page</a>] [<a href="https://openreview.net/forum?id=rI6lxIX0uX">openreview</a>]</h3>

<img width="100%" src="https://github.com/huiwon-jang/RSP/assets/69646951/7ee0066f-f1a5-4db1-84b5-8ccb3862475a"/>


### 1. Environment setup
- We note that torch version >2.0 may work, but `conda install` with below version is recommended.
```bash
conda create -n rsp python=3.9.12 -y
conda activate rsp
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 2. Dataset

#### Dataset download
```bash
sh data_preprocessing/download.sh
sh data_preprocessing/extract.sh
```
- We assume the root directory for the data: `$DATA_ROOT = /data/kinetics400`.
- If you want to change the root directory, please change `root_dl` of `download.sh` and `extract.sh`.

#### Dataset pre-processing
- We resize the data into 256x256 for the efficient loading while training.
```bash
python data_preprocessing/make_256scale.py --datadir $DATA_ROOT
```
- We additionally provide the code to filter out several not-working videos.
```bash
python data_preprocessing/make_labels.py --datadir $DATA_ROOT --filedir train2
```

#### Kinetics-400
```
/data/kinetics400
|-- train2
    |-- abseiling
        |-- xx.mp4
        |-- ...
    |-- air_drumming
        |-- xx.mp4
        |-- ...
    |-- ...
|-- labels
    |-- label_full_1.0.pickle
```

### 3. Pre-training RSP on Kinetics-400
- Note that `[N_NODE] x [BATCH_SIZE_PER_GPU] x [ACCUM_ITER] = 1536` to reproduce our results.
- Default: `[DATA_PATH]=/data/kinetics400 `
```bash
python -m torch.distributed.launch --nproc_per_node=[N_NODE] main_pretrain_rsp.py \
    --batch_size [BATCH_SIZE_PER_GPU] \
    --accum_iter [ACCUM_ITER] \
    --model rsp_vit_small_patch16 \
    --epochs 400 \
    --warmup_epochs 40 \
    --data_path [DATA_PATH] \
    --log_dir [LOG_DIR] \
    --output_dir [LOG_DIR] \
    --norm_pix_loss \
    --repeated_sampling 2
```

### 4. Evaluation
We provide the checkpoint in the below:
- ViT-S/16 400 epochs: [[link](https://drive.google.com/file/d/1OZW7UWOLhBxjXukn8CAuCSA9mCV_d9R2/view?usp=drive_link)]
- ViT-B/16 400 epochs: [[link](https://drive.google.com/file/d/1GsA9h1w4RZ4unpJ2td1GaI_yfZ3vB2Rm/view?usp=sharing)]

#### Video Label Propagation
The evaluation code is mainly built upon [Dino](https://github.com/facebookresearch/dino).
1. **DAVIS 2017 video object segmentation**
- **Step 1: Dataset preparation**

We note that the default root path is `[DATA_ROOT]=/data`.
```bash
sh data_preprocessing/eval/davis_download.sh
python data_preprocessing/eval/davis_preprocessing.py --data_root [DATA_ROOT]
```
We resize DAVIS of 480x(?) to 480x880 for a natural evaluation with patches.
```
[DATA_ROOT]/DAVIS_480_880
|-- Annotations/480p
    |-- bear
        |-- 00000.png
        |-- ...
    |-- ...
|-- ImageSets/2017/val.txt
|-- JPEGImages/480p
    |-- bear
        |-- 00000.jpg
        |-- ...
    |-- ...
```

- **Step 2: Video object segmentation**
```bash
python eval_video_segmentation_davis.py \
    --finetune [LOG_DIR]/checkpoint-199.pth \
    --output_dir [LOG_DIR]/davis_seg \
    --data_path [DATA_ROOT]/DAVIS_480_880 \
    --topk 7 --size_mask_neighborhood 30 --n_last_frames 30 \
    --model vit_small
```

- **Step 3: Evaluation the obtained segmentation**
```bash
git clone https://github.com/davisvideochallenge/davis2017-evaluation
python ./davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --results_path [LOG_DIR]/davis_seg \
    --davis_path [DATA_ROOT]/DAVIS_480_880
```

2. **JHMDB pose tracking**
TBD

3. **VIP video part segmentation**
TBD

#### Vision-based Robot Learning
1. CortexBench
TBD

2. RLBench
TBD

3. Franka Kitchen
TBD
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

# Evaluation RSP on Robot Learning tasks

### 1. CortexBench
The evaluation code is mainly built upon [eai-vc](https://github.com/facebookresearch/eai-vc).

#### 1.1. Dataset preparation
Please see the `eai-vc-custom/coretexbench/DATASETS.md` to install the dataset in the correct directory as follows.

- Adroit
```

```
- Metaworld

- DMControl

- Trifinger
```
eai-vc-custom/cortexbench/trifinger_vc/assets/bc_demos/data/trifinger-demos
|-- move
    |-- demo-0000
        |-- dts-0p02
            |-- downsample.pth
            |-- rgb_image_60.gif
        |-- dts-0p4
            |-- downsample.pth
            |-- rgb_image_60.gif
        |-- demo-0000.npz
    |-- ...
|-- reach
    |-- demo-0000
        |-- dts-0p02
            |-- downsample.pth
            |-- rgb_image_60.gif
        |-- dts-0p4
            |-- downsample.pth
            |-- rgb_image_60.gif
        |-- demo-0000.npz
    |-- ...
```

#### 1.2. Model configuration
- The model configuration is handled in `eai-vc-custom/vc_models/src/vc_models/conf/model/rsp_vits16.yaml`.
- You need to replace `[LOG_DIR]` in `model/checkpoint_path` properly to evaluate our model.
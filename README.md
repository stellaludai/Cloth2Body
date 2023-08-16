# Cloth2Body
Code and data repo for ICCV 2023 paper Cloth2Body


# Pipeline
<figure>
    <img src="./assets/pipeline.png" alt="pipeline">
    <figcaption 
    style="text-align: center"> 
    Pipeline 
    </figcaption>
</figure>

# Data Preprocessing
**DeepFashion2-SMPL dataset:** 

https://drive.google.com/file/d/1y0Qm_FIYMsmpvrOzhI48EFJvJ8yQl0M0/view?usp=drive_link

**AGORA-CLOTH dataset:**

Given [AGORA's license](https://agora.is.tue.mpg.de/license.html), we are not directly releasing the images. Instead, we release the code of processing the AGORA dataset, as well as the annotations we run on the images.

# Training
## Train pose regressor
requirements:
clothing image
annotation file
```python
python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=12345 \
    tools/train.py \
    configs/hybrik/resnet34_hybrik_agoracloth.py \
    --work-dir=experiments \
    --launcher pytorch \
    --no-validate
```

## Train shape regressor
requirements:
clothing images
landmarks
```python
python cloth2smpl/res_cvae_shape/trainer.py
```


# Evaluation
requirements:
```python
python -m torch.distributed.launch tools/test.py \
    configs/hybrik/resnet34_hybrik_agoracloth.py \
    --work-dir=experiments \
    --metrics pa-mpjpe mpjpe 2dkpe shape_pve \
    --eval_dataset df2
```
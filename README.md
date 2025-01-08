# Patch-Based Spatio-Temporal Deformable Attention BiRNN for Video Deblurring 

[Huicong Zhang](https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&gmla=AETOMgEHtB1sAOmB8EMhprsRACCsD_wLbTGpnaBrkyshm-oVsQtYAjL8q9BRZI6gOiD6nQZSg_urpfJV1FgXa1iGGU6rPo0&user=ASaPjIgAAAAJ)<sup>1</sup>, [Haozhe Xie](https://haozhexie.com)<sup>2</sup>, [Shengping Zhang](https://scholar.google.com/citations?user=hMNsT8sAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>,
[Hongxun Yao](https://scholar.google.com/citations?user=aOMFNFsAAAAJ)<sup>1</sup>

<sup>1</sup>Harbin Institute of Technology, <sup>2</sup>S-Lab, Nanyang Technological University

## Changelogs
- [2024/12/20] The training and testing code are released.

## Datasets

We use the [GoPro](https://github.com/SeungjunNah/DeepDeblur_release), [BSD](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view?usp=sharing) and [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/) datasets in our experiments, which are available below:

- [GoPro](https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8?usp=sharing)
- [DVD](https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8?usp=sharing)
- [BSD](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view?usp=sharing)

You could download the zip file and then extract it to the [datasets](datasets) folder. 

## Pretrained Models

You could download the pretrained model from [here](https://drive.google.com/drive/folders/15v9J7uPli2f5Q0Ce4hNQPrimHjdpwMQe?usp=sharing) and put the weights in [model_zoos](model_zoos). 

## Dataset Organization Form
If you prepare your own dataset, please follow the following form like GOPRO/DVD/BSD:
```
|--dataset  
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
        	:
        |--video n
```
 

## Prerequisites
#### Clone the Code Repository

```
git clone https://github.com/huicongzhang/STDANetPP.git
```
### Install Denpendencies

```
conda create -n STDANetPP python=3.8
conda activate STDANetPP
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim
mim install mmcv-full
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop
cd basicsr/ops/mda
python setup.py build install
```

## Test
To test STDANetPP, you can simply use the following commands:

GoPro dataset
```
python basicsr/test.py -opt options/test/STDANetPP_gopro.yml
```

DVD dataset
```
python basicsr/test.py -opt options/test/STDANetPP_dvd.yml
```

BSD(1ms_8ms) dataset
```
python basicsr/test.py -opt options/test/STDANetPP_bsd1.yml
```

BSD(2ms_16ms) dataset
```
python basicsr/test.py -opt options/test/STDANetPP_bsd2.yml
```

BSD(3ms_24ms) dataset
```
python basicsr/test.py -opt options/test/STDANetPP_bsd3.yml
```



## Train
To train STDANetPP, you can simply use the following commands:

GoPro dataset
```
scripts/dist_train.sh 2 options/train/STDANetPP_gopro.yml
```

DVD dataset
```
scripts/dist_train.sh 2 options/train/STDANetPP_dvd.yml
```

BSD(1ms_8ms) dataset
```
scripts/dist_train.sh 2 options/train/STDANetPP_bsd1.yml
```

BSD(2ms_16ms) dataset
```
scripts/dist_train.sh 2 options/train/STDANetPP_bsd2.yml
```

BSD(3ms_24ms) dataset
```
scripts/dist_train.sh 2 options/train/STDANetPP_bsd3.yml
```


## License

This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [ProPainter](https://github.com/sczhou/ProPainter) and [RVRT](https://github.com/JingyunLiang/RVRT). 


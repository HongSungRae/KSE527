# KSE527
> Team Project of KSE527, Spring 2022

## 1. Introduction
(대충 소개)

## 2. How To Use?
### 2.1 Environment Setting
1. Your PC or server should have a GPU and cuda setting.
2. Clone this repo to your environment. : ```git clone <this_repo>```
3. Check the requirements. : ```pip install -r requirements.txt```

### 2.2 Dataset Download
1. Download ```fma_small``` and ```fma_medium``` from the [LINK](https://github.com/mdeff/fma).
2. Put ```fma_small``` and ```fma_medium``` to ```./data``` folder.
3. Download ```ffmpeg.exe``` from [LINK](https://www.ffmpeg.org/download.html) and put it to ```./```. For Koreans [this page](https://m.blog.naver.com/chandong83/222095346417) can help you.
---
*If your Local environment is as below, you're all set!*
---
```
<KSE527>
    ├ <data>
        ├ <fma_medium>
        ├ <fma_small>
        ├ genre_dic_medium.npy
        └ genre_dic_small.npy
    ├ <models>
        ├ simsiam.py
        └ tester.py
    ├ augmentation.py
    ├ dataset.py
    ├ ffmpeg.exe
    ├ metrics.py
    ├ parameters.py
    ├ preprocessing.py
    ├ supervise.py
    ├ train_simsiam.py
    ├ tansfer.py
    ├ utils.py
    ├ LICENSE
    ├ requirements.txt
    └ README.md                           
```

### 2.3 Data Pre-processing
- You need to convert all ```.wma``` files to ```.npy```(because of processing speed).
1. For Fma_medium
```
python preprocessing.py --size medium
```

2. For Fma_small
```
python preprocessing.py --size small
```

### 2.4 PreTraining -> Transfer Learning -> Inference
1. PreTraining
```
```
2. Transfer learning to downstream task
```
```
3. Inference and performance
```
```

## 3. Results
### 3.1 Image Augmentation For Audio Spectrogram
![Augmentation](https://github.com/HongSungRae/KSE527/blob/main/archive/augmentation.jpg?raw=true)

### 3.2 Experiment Results
![](https://github.com/HongSungRae/KSE527/blob/main/archive/table1.png?raw=true)
![](https://github.com/HongSungRae/KSE527/blob/main/archive/table2.png?raw=true)

### 3.3 LaTeX Formed Report


## Acknowledgement in Korean
- 플젝 잘 끝내서 모두에게 감사합니다.
- 모두 건강하세요~ 💪 :smile: :smile:

## Licence
We follow ```MIT LICENCE```

## Contact
- TaeMi Kim : taemi_kim@kaist.ac.kr
- SungRae Hong : sun.hong@kaist.ac.kr
- Sol Lee : leesol4553@kaist.ac.kr

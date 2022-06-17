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
2. ```./data``` 에 다운로드 받은 두 폴더를 모두 넣어주세요.
3. ```ffmpeg.exe```를 링크에서 다운받아서 LOCAL ```./```에 넣어주세요.
---
**여러분의 Local 환경이 아래 그림과 같다면 모든 준비가 끝났습니다!**
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


### 2.3 Training + Inference

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

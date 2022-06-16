import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from tsne_torch import TorchTSNE as TSNE

#local
from parameters import *




def get_tSNE(exp_path,dataloader,iteration=30):
    '''
    exp_path : encoder가 있는 위치 폴
    iter : 몇번 반복해서 들려줄 것인지
    '''
    total_vector = torch.zeros((len(dataloader),2048))

    model = torch.load(f'./exp/{exp_path}/transfer.pth')
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels).cuda()
    to_db = torchaudio.transforms.AmplitudeToDB().cuda()
    spec_bn = nn.BatchNorm2d(1).cuda()

    for i in tqdm(range(iter),desc='음악 듣는중...'):
        idx = 0
        for idx, (audio,label) in enumerate(dataloader):
            batch_size = audio.shape[0]
            audio = audio.float().cuda() # (B, 22050*5)
            audio = spec(audio)
            audio = to_db(audio)
            audio = audio.unsqueeze(1)
            audio = spec_bn(audio)
            latent_vector = model.encoder(audio)
            latent_vector = torch.squeeze(latent_vector,dim=0)
            total_vector[idx:idx+batch_size,:] += latent_vector.detach().cpu()/iteration
            
        # vector = total_vector.detach().cpu().clone() # [50, 2048]

        

    try:
        X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(total_vector)  # returns shape (n_samples, 2)
    except:
        # 오류나는 코드가 library에서 출발한다. 코드 까보니까 Index 설정이 뭐 하나 잘못되어있다
        dummy = torch.zeros(50,1)
        vector = torch.cat((vector,dummy),dim=-1)
        X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(total_vector)  # returns shape (n_samples, 2)
    pass
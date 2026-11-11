"""PyTorch sequence architectures for the true first-3.6-V discharge-prefix study."""
from __future__ import annotations
import torch
from torch import nn

CONFIGS = {
    'LSTM': {
        'L1': dict(layers=1, hidden=32, dropout=0., lr=1e-3),
        'L2': dict(layers=1, hidden=64, dropout=0., lr=1e-3),
        'L3': dict(layers=2, hidden=64, dropout=.1, lr=3e-4),
    },
    'Transformer': {
        'T1': dict(layers=1, hidden=32, heads=4, ffn=64, dropout=0., lr=1e-3),
        'T2': dict(layers=1, hidden=64, heads=4, ffn=128, dropout=0., lr=1e-3),
        'T3': dict(layers=2, hidden=64, heads=4, ffn=128, dropout=.1, lr=3e-4),
    },
}


class LSTMRegressor(nn.Module):
    def __init__(self, config: dict, with_age: bool):
        super().__init__()
        h=config['hidden']
        self.lstm=nn.LSTM(input_size=3,hidden_size=h,num_layers=config['layers'],
                          dropout=config['dropout'] if config['layers']>1 else 0.,batch_first=True)
        self.with_age=with_age
        self.head=nn.Sequential(nn.Linear(h+int(with_age),32),nn.ReLU(),nn.Linear(32,1))

    def forward(self, x, age=None):
        seq,(hidden,_)=self.lstm(x)
        pooled=hidden[-1]
        if self.with_age:
            if age is None:raise ValueError('Age is required')
            pooled=torch.cat([pooled,age.reshape(-1,1)],dim=1)
        return self.head(pooled).flatten()


class TransformerRegressor(nn.Module):
    def __init__(self, config:dict, with_age:bool, length:int=96):
        super().__init__()
        h=config['hidden']
        self.projection=nn.Linear(3,h)
        self.position=nn.Parameter(torch.zeros(1,length,h))
        nn.init.normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=h,nhead=config['heads'],
                    dim_feedforward=config['ffn'],dropout=config['dropout'],
                    activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=config['layers'])
        self.norm=nn.LayerNorm(h)
        self.with_age=with_age
        self.head=nn.Sequential(nn.Linear(h+int(with_age),32),nn.GELU(),nn.Linear(32,1))

    def forward(self,x,age=None):
        z=self.norm(self.encoder(self.projection(x)+self.position[:,:x.shape[1],:]))
        pooled=z.mean(dim=1)  # exactly the 96 encoded DISCHARGE prefix samples
        if self.with_age:
            if age is None:raise ValueError('Age is required')
            pooled=torch.cat([pooled,age.reshape(-1,1)],dim=1)
        return self.head(pooled).flatten()


def create_model(family:str,config:dict,with_age:bool,length:int=96):
    if family=='LSTM':return LSTMRegressor(config,with_age)
    if family=='Transformer':return TransformerRegressor(config,with_age,length)
    raise ValueError(f'Unknown neural family: {family}')

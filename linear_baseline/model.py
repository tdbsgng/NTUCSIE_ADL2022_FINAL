from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ref: https://ithelp.ithome.com.tw/articles/10268146

class MultiLabelCls(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_class: int,
    ) -> None:
        super(MultiLabelCls, self).__init__()
        # TODO: model architecture
        # raise NotImplementedError
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.enc = nn.Linear(input_size, self.hidden_size)
        self.dp1 = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.dp2 = nn.Dropout(dropout)
        #
        self.enc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act1 = nn.ReLU()
        self.dp3 = nn.Dropout(dropout)
        self.dp4 = nn.Dropout(dropout)
        #
        '''
        self.enc2 = nn.Linear(self.hidden_size//2, self.hidden_size//4)
        self.act2 = nn.ReLU()
        self.dp5 = nn.Dropout(dropout)
        self.dp6 = nn.Dropout(dropout)
        '''
        #
        self.dec = nn.Linear(self.hidden_size, num_class)
    

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        # raise NotImplementedError
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = batch["one_hot_user_data"]

        out = self.enc(x)
        out = self.dp1(out)
        out = self.act(out)
        out = self.dp2(out)
        #
        out = self.enc1(out)
        out = self.dp3(out)
        out = self.act1(out)
        out = self.dp4(out)
        #
        '''
        out = self.enc2(out)
        out = self.dp5(out)
        out = self.act2(out)
        out = self.dp6(out)
        '''
        #
        out = self.dec(out)

        return out
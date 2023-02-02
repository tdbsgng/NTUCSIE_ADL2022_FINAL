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
        self.hid = nn.Linear(self.hidden_size, self.hidden_size)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
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
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.dp3(out)
        out = self.dec(out)

        return out

class MultiLabelCls_subgroup(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_class: int,
    ) -> None:
        super(MultiLabelCls_subgroup, self).__init__()
        # TODO: model architecture
        # raise NotImplementedError
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.enc = nn.Linear(input_size, self.hidden_size)
        self.dp1 = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.hid = nn.Linear(self.hidden_size, self.hidden_size)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
        self.dec = nn.Linear(self.hidden_size, num_class)
    

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        # raise NotImplementedError
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = torch.cat([batch["one_hot_user_data"], batch["one_hot_course_data"]], dim=2) 
        out = self.enc(x)
        out = self.dp1(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.dp3(out)
        out = self.dec(out)

        return out

class MultiLabelCls_3(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_class: int,
    ) -> None:
        super(MultiLabelCls_3, self).__init__()
        # TODO: model architecture
        # raise NotImplementedError
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.enc = nn.Linear(input_size, self.hidden_size)
        self.dp1 = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.hid = nn.Linear(self.hidden_size, self.hidden_size)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
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
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.dp3(out)
        out = self.dec(out)

        return out

class MultiLabelCls_subgroup_2(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_class: int,
    ) -> None:
        super(MultiLabelCls_subgroup_2, self).__init__()
        # TODO: model architecture
        # raise NotImplementedError
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.enc = nn.Linear(input_size, self.hidden_size)
        self.dp1 = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.hid = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.hid2 = nn.Linear(int(self.hidden_size/2), int(self.hidden_size/4))
        self.hid3 = nn.Linear(int(self.hidden_size/4), int(self.hidden_size/8))
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
        self.dec = nn.Linear(int(self.hidden_size/8), num_class)
    

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        # raise NotImplementedError
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = torch.cat([batch["one_hot_user_data"], batch["one_hot_course_data"]], dim=2) 
        out = self.enc(x)
        out = self.dp1(out)
        out = self.act(out)
        out = self.hid(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid2(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.hid3(out)
        out = self.dp2(out)
        out = self.act(out)
        out = self.dp3(out)
        out = self.dec(out)

        return out
from torch import nn


def getLSTM(input_size, hidden_size, bidirectional):
    return nn.LSTM(input_size= input_size,
                                    hidden_size=hidden_size,
                                    num_layers=4,
                                    batch_first=True,
                                    dropout=0.1,
                                    bidirectional=bidirectional)
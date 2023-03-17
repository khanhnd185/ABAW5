import math
import torch.nn as nn

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()

class  Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='none', bn=False, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        if activation == 'tanh':
            self.ac = nn.Tanh()
        elif activation == 'softmax':
            self.ac = nn.Softmax()
        elif activation == 'sigmoid':
            self.ac = nn.Sigmoid()
        elif activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        else:
            self.ac = nn.Identity()
        
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        else:
            self.bn = nn.Identity()

        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class Attention(nn.Module):
    def __init__(self, encoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att = self.full_att(self.relu(att1)).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class MLP(nn.Module):
    def __init__(self, num_class=8, feature_size=1288):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            Dense(feature_size, 512, activation='relu', drop=0.2),
            Dense(512, 64, activation='relu', drop=0.2),
            Dense(64, num_class, activation='softmax'),
        )

    def forward(self, x):
        x = self.model(x)
        return x
import torch
import math
import torch.nn as nn

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()

class  Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', bn=False, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        if activation == 'tanh':
            self.ac = nn.Tanh()
        elif activation == 'softmax':
            self.ac = nn.Softmax()
        elif activation == 'sigmoid':
            self.ac = nn.Sigmoid()
        else:
            self.ac = nn.ReLU(inplace=True)
        
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

class AFER(nn.Module):
    def __init__(self, num_class=8, train_backbone=False):
        super(AFER, self).__init__()

        self.backbone = torch.load('../MTL-ABAW3/model/enet_b0_8_best_vgaf.pt')
        self.pooling = self.backbone.global_pool
        self.classifier = self.backbone.classifier[0]

        dim_score, dim_embedding = self.classifier.weight.data.shape
        feature_size = dim_score + dim_embedding

        self.attention = Attention(dim_embedding, 128)
        self.ex = Dense(feature_size, num_class, activation='softmax', drop=0.2)
        self.ex = nn.Sequential(
            Dense(feature_size, 512, activation='relu', drop=0.2),
            Dense(512, 64, activation='relu', drop=0.2),
            Dense(64, num_class, activation='softmax'),
        )

        self.backbone.global_pool = torch.nn.Identity()
        self.backbone.classifier  = torch.nn.Identity()

        if train_backbone == False:
            self.freeze_backbone()

    def forward(self, x):
        x = self.backbone(x)
        b, c, h, w = x.shape
        embedding = x.view(b, c, -1).permute(0, 2, 1)
        ex, a_ex = self.attention(embedding)
        score = self.classifier(self.pooling(x))
        ex = torch.cat((ex, score), dim=-1)
        ex = self.ex(ex)
        return ex

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False


class MLP(nn.Module):
    def __init__(self, num_class=8, feature_size=1288):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            Dense(feature_size, 640, activation='relu', drop=0.2),
            Dense(640, 80, activation='relu', drop=0.2),
            Dense(80, num_class, activation='softmax'),
        )

    def forward(self, x):
        x = self.model(x)
        return x
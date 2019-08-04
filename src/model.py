import torch as th
import torch.nn as nn
from torchvision.models import resnet34


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet34(pretrained=True)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        B, C = x.size()[:2]
        x = x.view(B, C, -1).transpose(1, 2)

        return x


class SoftAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, x, h):
        """
        Parameters
        ==========
        x: B x L x C
        h: B x C

        Returns
        =======
        output: B x C
        """
        e = th.einsum('bij,bj->bi', x, h)
        a = self.softmax(e)
        z = th.einsum('bi,bij->bj', a, x)
        return z, a


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, vocab_size, hidden_size, bgn, attention):
        super().__init__()
        self.register_buffer("bgn", th.LongTensor([bgn]))
        self.f_c = nn.Linear(input_size, hidden_size)
        self.f_h = nn.Linear(input_size, hidden_size)
        self.vocab = nn.Embedding(vocab_size, embedding_size)
        self.attn = attention
        self.lstm = nn.LSTMCell(embedding_size + input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.LogSoftmax(1)
        )

    def get_init_hidden(self, x):
        """
        x: B x L x C
        """
        avg = x.mean(1)
        c = self.f_c(avg)
        h = self.f_h(avg)
        return h, c

    def forward(self, x, hidden, word):
        h, c = hidden
        Ey = self.vocab(word)
        z, a = self.attn(x, h)  # B x C
        lstm_x = th.cat([Ey, z], 1)
        h, c = self.lstm(lstm_x, (h, c))
        return (h, c), self.fc(self.dropout(h)), a


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, vocab_size, hidden_size, bgn, attention, searcher):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(input_size, embedding_size, vocab_size, hidden_size, bgn, attention)
        self.searcher = searcher

    def partial(self, input):
        x = self.encoder(input)
        init_hidden = self.decoder.get_init_hidden(x)
        init_word = self.decoder.bgn.expand(input.size(0))

        def _partial(hidden, word):
            return self.decoder(x, hidden, word)

        return _partial, init_hidden, init_word

    def forward(self, input, max_depth):
        return self.searcher.apply_batch(input, max_depth)

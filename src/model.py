import numpy as np
import torch as th
import torch.nn as nn
from torchvision.models import vgg19_bn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = vgg19_bn(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)

        B, C = x.size()[:2]
        x = x.view(B, C, -1).transpose(1, 2)

        return x


class SoftAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.f_beta = nn.Sequential(
            nn.Linear(decoder_dim, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()
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
        att1 = self.encoder_att(x)
        att2 = self.decoder_att(h)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        z = th.einsum('bi,bij->bj', alpha, x)
        gate_beta = self.f_beta(h)
        return gate_beta * z, alpha


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, vocab_size, hidden_size, attn_size, bgn, attention):
        super().__init__()
        self.embedding_size = embedding_size
        self.register_buffer("bgn", th.LongTensor([bgn]))
        self.f_c = nn.Linear(input_size, hidden_size)
        self.f_h = nn.Linear(input_size, hidden_size)
        self.vocab = nn.Embedding(vocab_size, embedding_size)
        self.attn = attention(input_size, hidden_size, attn_size)
        self.lstm = nn.LSTMCell(embedding_size + input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size + input_size + hidden_size, vocab_size),
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

    def load_pretrained_embedding(self, model, vocabulary):
        filename = f".cache/embedding-{model}.npy"
        try:
            matrix = np.load(filename)
        except Exception:
            matrix = self._load_gensim_embedding(model, vocabulary)
            np.save(filename, matrix)
        state = {"weight": th.tensor(matrix).float()}
        self.vocab.load_state_dict(state)
        print("loaded pretrained embeddings")

    def _load_gensim_embedding(self, model, vocabulary):
        import gensim.downloader as api
        print(f"loading gensim model {model}")
        model = api.load(model)
        matrix = np.empty([len(vocabulary), self.embedding_size])
        for idx, word in vocabulary.idx2word.items():
            try:
                array = model.get_vector(word)
            except KeyError:
                array = np.random.uniform(-0.1, 0.1, self.embedding_size)
            matrix[idx] = array
        return matrix

    def forward(self, x, hidden, word):
        h, c = hidden
        Ey = self.vocab(word)
        z, a = self.attn(x, h)  # B x C
        lstm_x = th.cat([Ey, z], 1)
        h, c = self.lstm(lstm_x, (h, c))
        x = th.cat([Ey, z, h], 1)
        return (h, c), self.fc(self.dropout(x)), a


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, vocab_size, hidden_size, attn_size, bgn, attention):
        super().__init__()
        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = Decoder(input_size, embedding_size, vocab_size, hidden_size, attn_size, bgn, attention)

    def partial(self, input):
        x = self.encoder(input)
        init_hidden = self.decoder.get_init_hidden(x)
        init_word = self.decoder.bgn.expand(input.size(0))

        def _partial(hidden, word):
            return self.decoder(x, hidden, word)

        return _partial, init_hidden, init_word

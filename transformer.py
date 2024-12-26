# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: transformer2
#     language: python
#     name: python3
# ---

import torch.nn as nn
import torch
from torch.nn.functional import log_softmax
import copy
import math
import time
from torch.optim.lr_scheduler import LambdaLR


# +
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


# -


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        y = self.proj(x)
        # apply log(softmax(x)) along the last dimension
        return log_softmax(y, dim=-1)


# it will normalize the d_model dimension which is the last dimension
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        avg = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - avg) / (std + self.eps) + self.b_2


# clone the layer for N times
def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Sublayer(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # from paper
    # We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized
    # That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.
    # my comment
    # the output of the previous sublayer is first normalized and then used as input to the next sublayer
    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


# # Attention Layer


def subsequent_mask(size, device=torch.device("cpu")):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # torch.triu will only keep the upper triangle part of the matrix and the other elements will be set to 0.
    # diagonal will control how much to upper triangle keep
    subsequent_mask = torch.triu(
        torch.ones(attn_shape, device=device), diagonal=1
    ).type(torch.uint8)
    return subsequent_mask == 0


subsequent_mask(10)

torch.triu(torch.ones(5, 5)).type(torch.uint8)

torch.triu(torch.ones(5, 5), diagonal=1).type(torch.uint8)


#
# In practice, we compute the attention function on a set of queries
# simultaneously, packed together into a matrix $Q$.  The keys and
# values are also packed together into matrices $K$ and $V$.  We
# compute the matrix of outputs as:
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, torch.transpose(k, -2, -1)) / math.sqrt(d_k)
    # mask is of size (seq_length, seq_length), so not related to d_model
    # because it will mask out the attention between different token in the sequence
    if mask is not None:
        # fill with a very small negative number
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn


#
# Multi-head attention allows the model to jointly attend to
# information from different representation subspaces at different
# positions. With a single attention head, averaging inhibits this.
#
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# Where the projections are parameter matrices $W^Q_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in
# \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in
# \mathbb{R}^{hd_v \times d_{\text{model}}}$.
#
# In this work we employ $h=8$ parallel attention layers, or
# heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$. Due
# to the reduced dimension of each head, the total computational cost
# is similar to that of single-head attention with full
# dimensionality.


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        # the projected dimension of each attention head
        # and assume d_v = d_k
        self.d_k = d_model // heads
        self.heads = heads
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.linears = clone(nn.Linear(d_model, d_model), 4)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            # the original mask is of shape(batch_size, seq_length, seq_length)
            # to apply the mask to all heads, we need to add 1 dimension to the corresponding head dimension of the attention
            mask = mask.unsqueeze(1)
        # linear projection of q, k, v in batch
        q, k, v = [lin(x) for lin, x in zip(self.linears, (q, k, v))]
        # make multi-head
        nbatches = q.size(0)
        q, k, v = [
            x.view(nbatches, -1, self.heads, self.d_k) for x in (q, k, v)
        ]
        # swap the sequence length dimension and head dimension
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        # apply attention to each head
        x, self.attn = attention(q, k, v, mask, self.dropout)
        # concatenate each heads
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.heads * self.d_k)
        )
        del q
        del k
        del v
        # apply linear transformation
        res = self.linears[-1](x)
        return res


# # Position-wise Feed-Forward Networks

# In addition to attention sublayers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between
#
# $$
# \mathrm{FFN}(x) =
#     \mathrm{max}(0, xW_1 + b_1)W_2 + b_2
# $$
#
# While the linear transformations are the same across different
# positions, they use different parameters from layer to
# layer. Another way of describing this is as two convolutions with
# kernel size 1.  The dimensionality of input and output is
# $d_{\text{model}}=512$, and the inner-layer has dimensionality
# $d_{ff}=2048$.


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = self.linear2(self.dropout(self.linear1(x).relu()))
        return res


# # Embedding and Softmax

# Similarly to other sequence transduction models, we use learned
# embeddings to convert the input tokens and output tokens to vectors
# of dimension $d_{\text{model}}$.  We also use the usual learned
# linear transformation and softmax function to convert the decoder
# output to predicted next-token probabilities.  In our model, we
# share the same weight matrix between the two embedding layers and
# the pre-softmax linear transformation, similar to
# [(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers,
# we multiply those weights by $\sqrt{d_{\text{model}}}$.


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # return self.lut(x) * math.sqrt(self.d_model)
        return self.lut(x) * math.sqrt(self.d_model)


# # Positional Encoding
#
# Since our model contains no recurrence and no convolution, in order
# for the model to make use of the order of the sequence, we must
# inject some information about the relative or absolute position of
# the tokens in the sequence.  To this end, we add "positional
# encodings" to the input embeddings at the bottoms of the encoder and
# decoder stacks.  The positional encodings have the same dimension
# $d_{\text{model}}$ as the embeddings, so that the two can be summed.
# There are many choices of positional encodings, learned and fixed
# [(cite)](https://arxiv.org/pdf/1705.03122.pdf).
#
# In this work, we use sine and cosine functions of different frequencies:
#
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
#
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
#
# where $pos$ is the position and $i$ is the dimension.  That is, each
# dimension of the positional encoding corresponds to a sinusoid.  The
# wavelengths form a geometric progression from $2\pi$ to $10000 \cdot
# 2\pi$.  We chose this function because we hypothesized it would
# allow the model to easily learn to attend by relative positions,
# since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a
# linear function of $PE_{pos}$.
#
# In addition, we apply dropout to the sums of the embeddings and the
# positional encodings in both the encoder and decoder stacks.  For
# the base model, we use a rate of $P_{drop}=0.1$.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -math.log(10000) / d_model
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        # add one dimension for batch
        pe = pe.unsqueeze(0)
        # make pe part of the module attribute and in the model dict, but not trainable
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x is the embeddings
        num_embeddings = x.size(1)
        return self.dropout(x + self.pe[:, :num_embeddings])


# # Encoder
# The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head
# self-attention mechanism, and the second is a simple, position-wise
# fully connected feed-forward network.


class EncoderLayer(nn.Module):
    def __init__(self, attn, fnn, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.fnn = fnn
        self.sublayers = clone(Sublayer(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayers[1](x, self.fnn)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # the input was normalized at the beginning of each layer
        # so for the final output, we need to normalize
        return self.norm(x)


# # Decoder
#
# In addition to the two sub-layers in each encoder layer, the decoder
# inserts a third sub-layer, which performs multi-head attention over
# the output of the encoder stack.  Similar to the encoder, we employ
# residual connections around each of the sub-layers, followed by
# layer normalization.


class DecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, fnn, d_model, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.fnn = fnn
        self.d_model = d_model
        self.sublayers = clone(Sublayer(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](
            x, lambda x: self.cross_attn(x, memory, memory, src_mask)
        )
        return self.sublayers[2](x, self.fnn)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# # EncoderDecoder


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, generator, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# # Full Model


def make_model(
    src_vocab_size,
    tgt_vocab_size,
    N=6,
    d_model=512,
    d_ff=2028,
    heads=8,
    dropout=0.1,
):
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model=d_model, heads=heads)
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
    src_embedding = Embedding(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embedding = Embedding(d_model=d_model, vocab_size=tgt_vocab_size)
    pos_encoding = PositionalEncoding(d_model=d_model)

    encoder = Encoder(EncoderLayer(c(attn), c(ffn), d_model), N)
    decoder = Decoder(DecoderLayer(c(attn), c(attn), ffn, d_model), N)
    generator = Generator(d_model, tgt_vocab_size)
    src_embed = nn.Sequential(src_embedding, c(pos_encoding))
    tgt_embed = nn.Sequential(tgt_embedding, c(pos_encoding))

    model = EncoderDecoder(encoder, decoder, generator, src_embed, tgt_embed)

    # weight initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# # Inference


def inference():
    test_model = make_model(11, 11, 1)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, src_mask)
    targets = torch.zeros(1, 1).type_as(src)
    for _ in range(9):
        tgt_mask = subsequent_mask(targets.size(1))
        out = test_model.decode(targets, memory, src_mask, tgt_mask)
        probs = test_model.generator(out[:, -1])
        _, next_word = torch.max(probs, dim=1)
        next_word = next_word[0]
        targets = torch.cat(
            [targets, torch.empty(1, 1).type_as(src).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", targets)


for _ in range(10):
    inference()


# # Train


class Batch:
    """Object for holding a batch of data with mask during training"""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        # src is of shape [batch_size, seq_len]
        self.src = src
        # the shape is [batch_size, 1, seq_len]
        # the additional one dimension is to make it broadcastable to attention scores of shape [batch_size, seq_len, seq_len]
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_tgt_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()

    @staticmethod
    def make_tgt_mask(tgt, pad):
        "create a mask to hide both pad and subsequent tokens"
        # tgt_mask is of shape [batch_size, 1, seq_len]
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # sbusequent_mask is of shape [1, seq_len, seq_len]
        # so here it uses the broadcast to get the output of shape [batch_size, seq_len, seq_len]
        return tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)


# # Train loop


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# +
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    device=torch.device("cpu"),
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    start_time = time.time()
    total_loss = 0
    total_tokens = 0
    tokens = 0  # token sum for each logging round
    n_accum = 0  # number of weight updates

    for i, batch in enumerate(data_iter):
        batch.src = batch.src.to(device)
        batch.tgt = batch.tgt.to(device)
        batch.tgt_y = batch.tgt_y.to(device)
        batch.src_mask = batch.src_mask.to(device)
        batch.tgt_mask = batch.tgt_mask.to(device)
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # normalized the loss with batch.ntokens
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            # accumulate gradients
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                # weights update
                optimizer.step()
                # clear accumulated gradients
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            # adjust learning rate
            scheduler.step()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            print(
                f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss/batch.ntokens:6.2f}| Tokens / Sec: {tokens/elapsed:7.1f} | Learning Rate: {lr:6.1e}"
            )
            start_time = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# -

# ## Optimizer
#
# We used the Adam optimizer [(cite)](https://arxiv.org/abs/1412.6980)
# with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We
# varied the learning rate over the course of training, according to
# the formula:
#
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
#
# This corresponds to increasing the learning rate linearly for the
# first $warmup\_steps$ training steps, and decreasing it thereafter
# proportionally to the inverse square root of the step number.  We
# used $warmup\_steps=4000$.

# > Note: This part is very important. Need to train with this setup
# > of the model.


def learning_rate(step, model_size, warmup_steps, factor):
    if step == 0:
        step = 1
    return (
        factor
        * model_size ** (-0.5)
        * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


# # Label Smoothing Regularization


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # the first token is padding token
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# # Synthetic Data

# You can see that we need to construct the training data into tensors of batches


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


# # Loss


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


# # Greedy Decoding
#
# The temperature parameter plays a role here. We just need to sample `next_word` using a Bernoulli distribution of parameter temperature instead of max


def greedy_decode(
    model, src, src_mask, max_len, start_symbol, device=torch.device("cpu")
):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1, device=device, dtype=torch.int64).fill_(
        start_symbol
    )
    for i in range(max_len - 1):
        out = model.decode(
            ys, memory, src_mask, subsequent_mask(ys.size(1), device=device)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [
                ys,
                torch.zeros(1, 1, device=device)
                .type_as(src.data)
                .fill_(next_word),
            ],
            dim=1,
        )
    return ys


# # Example Training


# +
def example_simple_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run on device: {device}")
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: learning_rate(
            step,
            model_size=model.src_embed[0].d_model,
            warmup_steps=400,
            factor=1.0,
        ),
    )

    batch_size = 20
    for epoch in range(3):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 2),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            device,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            device,
            mode="eval",
        )[0]

    model.eval()
    src = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long, device=device
    )
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len, device=device)
    print(
        greedy_decode(
            model,
            src,
            src_mask,
            max_len=max_len,
            start_symbol=0,
            device=device,
        )
    )


# execute_example(example_simple_model)
# -

example_simple_model()

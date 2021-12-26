import torch
import torch.nn as nn
from torch import Tensor
import random
from typing import Tuple
import torch.nn.functional as F


# %%
# Reference: https://github.com/SethHWeidman/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.birnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, code_bat: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(code_bat))

        self.birnn.flatten_parameters()
        output, hidden = self.birnn(embedded)

        hidden = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        hidden = torch.tanh(hidden)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim))

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # decoder_hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        # Step 1: to enable feeding through "self.attn" pink box above, concatenate
        # `repeated_decoder_hidden` and `encoder_outputs`:
        # torch.cat((hidden, encoder_outputs), dim = 2) has shape
        # [batch_size, seq_len, enc_hid_dim * 2 + dec_hid_dim]

        # Step 2: feed through self.attn to end up with:
        # [batch_size, seq_len, attn_dim]

        # Step 3: feed through tanh

        energy = torch.tanh(
            self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2))
        )

        # energy = [batch size, src sent len, attn_dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, attn_dim, src sent len]

        # v = [attn_dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, attn_dim]

        # High level: energy a function of both encoder element outputs and most recent decoder hidden state,
        # of shape attn_dim x enc_seq_len for each observation
        # v, being 1 x attn_dim, transforms this into a vector of shape 1 x enc_seq_len for each observation
        # Then, we take the softmax over these to get the output of the attention function

        attention = torch.bmm(v, energy).squeeze(1)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: int,
        attention: nn.Module,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(
        self, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(
        self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden, encoder_outputs
        )

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        self.rnn.flatten_parameters()
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self, code_bat: Tensor, doc_bat: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:

        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        batch_size = code_bat.shape[1]
        max_len = doc_bat.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(code_bat)

        # first input to the decoder is the <sos> token
        output = doc_bat[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = doc_bat[t] if teacher_force else top1

        return outputs


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


NB_FEATURES = len(code_vocab)
OUTPUT_DIM = len(doc_vocab)

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(NB_FEATURES, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device)

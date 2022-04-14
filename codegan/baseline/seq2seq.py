from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pack_sequence
from .. import tokenize


def mask_target(target_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """
    bos, token, eos -> 1
    pad -> 0
    """
    mask = torch.ones_like(target_ids, dtype=torch.int64)

    for i, batch in enumerate(target_ids):
        try:
            index_of_eos = (batch == tokenize.eos_token_id).nonzero()[0]
        except IndexError:
            continue
        mask[i][index_of_eos + 1:] = 0

    target_ids[(1 - mask).bool()] = tokenize.pad_token_id
    mask = mask.to(target_ids.device)
    return target_ids, mask


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size)

        """
        nn.GRU:
        L(x) = W * x + b
        zt = L1(h_t-1, x_t)
        rt = L2(h_t-1, x_t)
        RNN: ht  = tan(      L3(h_t-1) + L4(x_t) )
        GRU: ht' = tan( rt * L3(h_t-1) + L4(x_t) )
             ht = (1-zt) * ht' + zt * h_t-1
            
        Input:
            input: [L x N x input_size]
            (Optional) h_0: [D x N x hidden_size]
        Return:
            hiddens: [L x N x hidden_size]
            last_hidden: [D x N x hidden_size]
        """
        self.rnn = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor):
        '''
        Inputs:
            src: [src_len, batch_size]

        Outputs:
            enc_output: [src_len, batch_size, hidden_size * num_directions], all hidden states
            hidden: [batch_size, dec_hid_dim], the last hidden state fed through a linear layer
        '''
        src = src.transpose(0, 1)  # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)  # embedded = [src_len, batch_size, embed_size]

        enc_output: Tensor  # [src_len, batch_size, hidden_size * num_directions]
        hidden: Tensor  # [num_directions, batch_size, hidden_size]
        enc_output, hidden = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)))

        return enc_output, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((hidden_size * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden: Tensor, enc_output: Tensor):
        """
        Inputs:
            hidden: [src_len=1, batch_size, dec_hid_dim]
            enc_output: [src_len, batch_size, enc_hid_dim * num_directions]

        Outputs:
            weight: [batch_size, src_len]
        """
        src_len = enc_output.size(0)

        # repeat decoder hidden state src_len times
        hidden = hidden.transpose(0, 1).repeat(1, src_len, 1)  # [..., src_len, ...]
        enc_output = enc_output.transpose(0, 1)  # [batch_size, src_len, ...]

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_output], dim=2)))

        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, enc_hid_dim, hidden_size, dropout, with_attention=True):
        super().__init__()
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        if with_attention:
            self.attention = Attention(enc_hid_dim, hidden_size)
        self.rnn = nn.GRU((enc_hid_dim * 2) + embed_size, hidden_size)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + hidden_size + embed_size, output_size)

        self.with_attention = with_attention

    def forward(self, tgt: Tensor, hidden: Tensor, enc_output: Tensor):
        """
        Inputs:
            tgt: [batch_size]
            hidden: [1, batch_size, hidden_size]
            enc_output: [src_len, batch_size, enc_hid_dim * 2]

        Outputs:
            pred: [batch_size, output_size]
            hidden: [1, batch_size, hidden_size]
        """
        tgt = tgt.unsqueeze(1)  # [batch_size, 1]

        tgt_embed: Tensor  # [1, batch_size, embed_size]
        tgt_embed = self.dropout(self.embedding(tgt)).transpose(0, 1)

        if self.with_attention:
            # Add attention
            attn = self.attention(hidden, enc_output).unsqueeze(1)  # [batch_size, 1, src_len]
            # [1, src_len] @ [src_len, enc_hid_dim * 2] -> [1, enc_hid_dim * 2]
            enc_output = torch.bmm(attn, enc_output.transpose(0, 1)).transpose(0, 1)  # [1, batch_size, enc_hid_dim * 2]
        else:
            enc_output = enc_output[-1:]

        # [1, batch_size, (enc_hid_dim * 2) + embed_size]
        rnn_input = torch.cat([tgt_embed, enc_output], dim=2)

        # GRU's initial hidden is the last hidden output of encoder fed into a linear and a tanh
        # dec_output: Tensor  # [L=1, batch_size, hidden_size]
        # hidden: Tensor  # [D=1, batch_size, hidden_size]
        dec_output, hidden = self.rnn(rnn_input, hidden)

        tgt_embed = tgt_embed.squeeze(0)  # [batch_size, embed_size]
        dec_output = dec_output.squeeze(0)  # [batch_size, hidden_size]
        enc_output = enc_output.squeeze(0)  # [batch_size, hidden_size]

        pred: Tensor = self.fc_out(torch.cat([dec_output, enc_output, tgt_embed], dim=1))  # [batch_size, output_dim]

        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length,
                 encoder_hidden_size, decoder_hidden_size,
                 encoder_dropout=0.5, decoder_dropout=0.5,
                 with_attention=True):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, encoder_hidden_size, decoder_hidden_size, encoder_dropout)
        self.decoder = Decoder(vocab_size, embed_size, encoder_hidden_size, decoder_hidden_size,
                               decoder_dropout, with_attention=with_attention)

        self.decoder.embedding.weight = self.encoder.embedding.weight

        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        target_ids: Tensor = None,
        target_mask: Tensor = None,
        teacher_forcing_ratio=0.5,
        *args, **kwargs
    ):
        if target_ids is not None:
            return self.get_loss(source_ids, source_mask, target_ids, target_mask, teacher_forcing_ratio)

        return self.predict(source_ids, source_mask)

    def get_loss(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
        teacher_forcing_ratio=0.5
    ):
        """
        Inputs:
            source_ids: [batch_size x src_max_len]
                        values: [id(bos), ..., id(eos), n * id(pad)]
            source_mask: [batch_size x src_max_len]
                        values: [   1,    ...,    1,    n * 0]
            target_ids: [batch_size x tgt_max_len]
                        values: [id(bos), ..., id(eos), m * id(pad)]
            target_mask: [batch_size x tgt_max_len]
                        values: [   1,    ...,    1,    m * 0]
            teacher_forcing_ratio: the probability to use teacher forcing

        Outputs:
            loss:
            loss_num:
        """
        _src = source_ids.transpose(0, 1)
        _tgt = target_ids.transpose(0, 1)

        tgt_len = _tgt.size(0)

        # tensor to store decoder outputs
        outputs = []

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output: Tensor  # [L, N, H*D]
        hidden: Tensor  # [N, H]
        enc_output, hidden = self.encoder(_src)

        # first input to the decoder is the <sos> tokens
        dec_input = _tgt[0]
        hidden = hidden.unsqueeze(0)  # [1, N, H]

        for t in range(1, tgt_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # dec_output: Tensor [batch_size, hidden_size]
            # hidden: Tensor # [1, batch_size, hidden_size]
            logits, hidden = self.decoder(dec_input, hidden, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs.append(logits)

            # get the highest predicted token from our predictions
            top1 = logits.argmax(1)

            # decide if we are going to use teacher forcing or not
            teacher_forcing = torch.rand(1) < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = _tgt[t] if teacher_forcing else top1

        outputs = torch.stack(outputs).transpose(0, 1)  # [batch_size, tgt_len, hidden_size]

        active_tgt = target_ids[..., 1:].reshape(-1)
        active_loss = target_mask[..., 1:].ne(0).reshape(-1)

        loss = F.cross_entropy(
            outputs.reshape(-1, self.vocab_size)[active_loss],
            active_tgt[active_loss],
            ignore_index=tokenize.pad_token_id,
            reduction="sum"
        )

        return loss, active_loss.sum()

    def predict(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
    ):
        """
        Inputs:
            src: [batch_size, src_len]
            teacher_forcing_ratio: the probability to use teacher forcing

        Outputs:
            outputs: [batch_size, tgt_len]
            mask: [batch_size, tgt_len]
        """
        src = source_ids.transpose(0, 1)

        # tensor to store decoder outputs
        outputs = []

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output: Tensor  # [L, N, H*D]
        hidden: Tensor  # [N, H]
        enc_output, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = torch.full((source_ids.size(0),), tokenize.bos_token_id, device=src.device)  # tgt[0]
        hidden = hidden.unsqueeze(0)  # [1, N, H]

        for _ in range(1, self.max_length):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # dec_output: Tensor [batch_size, hidden_size]
            # hidden: Tensor # [1, batch_size, hidden_size]
            dec_output, hidden = self.decoder(dec_input, hidden, enc_output)

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)  # [batch_size]

            outputs.append(top1)

            dec_input = top1

        target = torch.stack(outputs, 1)
        return mask_target(target)

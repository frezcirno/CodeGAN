import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from codegan.beam import Beam
import tokenizer


def mask_target(target_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """
    bos, token, eos -> 1
    pad -> 0
    """
    mask = torch.ones_like(target_ids, dtype=torch.int64)

    for i, batch in enumerate(target_ids):
        try:
            index_of_eos = (batch == tokenizer.eos_token_id).nonzero()[0]
        except IndexError:
            continue
        mask[i][index_of_eos + 1:] = 0

    target_ids[(1 - mask).bool()] = tokenizer.pad_token_id
    mask = mask.to(target_ids.device)
    return target_ids, mask


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """
        Generator model, robert + transformer

        Parameters:

        * `hidden_size` - size of hidden layer
        * `vocab_size` - size of vocab
        * `max_length`- max length of target for beam search.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_length: int,
        dropout=0.5,
        nhead=8,
        dim_feedforward=2048,
        beam_size=10
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=tokenizer.pad_token_id)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, 6)

        decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, 6)

        self.output = nn.Linear(hidden_size, vocab_size)

        # register_buffer() - 模型的常量参数, 不会被训练
        # 类似:
        # [[0,   -1e4, -1e4],
        #  [0,    0,   -1e4],
        #  [0,    0,   0   ]]
        self.register_buffer("bias", torch.triu(torch.full((2048, 2048), -1e4), diagonal=1))

    def forward(
        self,
        source_ids=None,
        source_mask=None,
        target_ids=None,
        target_mask=None,
        beam_search=False,
    ):
        if target_mask is not None:
            return self.get_loss(source_ids, source_mask, target_ids, target_mask)

        if beam_search:
            return self.beam_predict(source_ids, source_mask)

        return self.predict(source_ids, source_mask)

    def get_loss(self, source_ids, source_mask, target_ids, target_mask):
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

        Outputs:
            loss: average of all loss
            active_loss: the number of loss
            lm_logits: [batch_size x tgt_max_len x vocab_size]
        """

        src_emb = self.embedding(source_ids)
        src = self.pos_encoder(src_emb)
        memory = self.encoder(src)

        tgt_emb = self.embedding(target_ids)
        tgt = self.pos_encoder(tgt_emb)

        tgt_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]  # [tgt_len x tgt_len]
        dec_output = self.decoder(tgt, memory, tgt_mask, memory_key_padding_mask=(1 - source_mask).bool())

        logits = self.output(dec_output)

        # [batch_size x (tgt_max_len-1) x vocab_size]
        shift_logits = logits[:, :-1, :]

        # Eval bos_token is invalid
        # [(batch_size*(tgt_max_len-1)) ]
        active_target = target_ids[..., 1:].reshape(-1)
        active_loss = target_mask[..., 1:].ne(0).reshape(-1)

        # Flatten the tokens
        loss = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size)[active_loss],
            # [batch_size*(tgt_max_len-1) x vocab_size]
            active_target[active_loss],  # [batch_size*(tgt_max_len-1) ]  (remove bos_token)
            ignore_index=tokenizer.pad_token_id,
            reduction='sum',
        )

        # return loss, loss * active_loss.sum(), active_loss.sum()
        return loss, active_loss.sum()

    def predict(self, source_ids, source_mask):
        """
        Predict use greedy search
        <s> -> a
        <s>,a -> a,b
        <s>,a,b -> a,b,c
        ...
        <s>,...,y -> ...,y,z
        return a,...,z

        Inputs:
            source_ids: [batch_size x src_max_len]
            source_mask: [batch_size x src_max_len]

        Outputs:
            target_ids: [batch_size x max_length], with bos
            target_mask: [batch_size x max_length]
        """
        batch_size = source_ids.size(0)
        device = source_ids.device

        src_emb = self.embedding(source_ids)
        src = self.pos_encoder(src_emb)
        memory = self.encoder(src)
        memory_key_padding_mask = (1 - source_mask).bool()

        target_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.int64, device=device)

        # has_end = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(self.max_length - 1):
            # [batch_size x vocab_size] (value: probs)
            tgt_emb = self.embedding(target_ids)
            tgt = self.pos_encoder(tgt_emb)

            tgt_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]  # [tgt_len x tgt_len]
            dec_output = self.decoder(tgt, memory, tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
            last_output = dec_output[:, -1, :]
            last_logits = self.output(last_output)

            # out = F.log_softmax(last_logits, dim=1)
            out = last_logits

            pred = out.argmax(1)  # [batch_size ] (values: id)
            target_ids = torch.cat([target_ids, pred.unsqueeze(1)], 1)

        # target_ids: [batch_size x max_length]
        # [batch_size x max_length x vocab_size]

        return mask_target(target_ids)

    def beam_predict(self, source_ids, source_mask, _device=None):
        """
        Inputs:
            source_ids: [batch_size x src_max_len]
            source_mask: [batch_size x src_max_len]

        Outputs:
            [batch_size x max_length] (value: id), padded by pad_token_id
        """
        src_emb = self.embedding(source_ids)
        src = self.pos_encoder(src_emb)
        context = self.encoder(src)

        preds = []
        batch_size = source_ids.size(0)
        device = _device if _device else source_ids.device

        # Beam search for every sample
        for i in range(batch_size):
            beam = Beam(self.beam_size, source_ids.device)
            btarget_ids = beam.getCurrentState(source_ids.device)  # [beam_size x 1]
            # [batch_size x src_max_len x hidden_size]
            # -> [1 x src_max_len x hidden_size]
            # -> [beam_size x src_max_len x hidden_size]
            ctx = context[i: i + 1].repeat(self.beam_size, 1, 1)
            # [batch_size x src_max_len]
            # -> [1 x src_max_len]
            # -> [beam_size x src_max_len]
            context_mask = source_mask[i: i + 1, :].repeat(self.beam_size, 1)
            memory_key_padding_mask = (1 - context_mask).bool()
            # [beam_size x src_max_len]
            for _ in range(self.max_length):
                if beam.done():
                    break

                tgt_emb = self.embedding(btarget_ids)
                tgt = self.pos_encoder(tgt_emb)

                tgt_mask = self.bias[: btarget_ids.shape[1], : btarget_ids.shape[1]]  # [tgt_len x tgt_len]
                dec_output = self.decoder(tgt, ctx, tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
                last_output = dec_output[:, -1, :]
                last_logits = self.output(last_output)

                lm_logits = F.log_softmax(last_logits, dim=1)  # [beam_size x vocab_size] (value: probs)

                beam.advance(lm_logits.data)

                # generate a word
                # copy_() - similar to assign
                btarget_ids.data.copy_(
                    # index_select() - choose row(0) by indexes
                    btarget_ids.data.index_select(0, beam.getCurrentOrigin())
                )
                btarget_ids = torch.cat([btarget_ids, beam.getCurrentState(source_ids.device)], 1)
                # [beam_size x i]

            # [beam_size x [n]] (values: token_id)
            hyp = beam.getHyp(beam.getFinal())
            # truncate
            beam_preds = beam.buildTargetTokens(hyp)[: self.beam_size]
            # [beam_size x <=max_length]

            best = beam_preds[0]
            raw = [tokenizer.bos_token_id] + [x.item() for x in best]
            if len(raw) > self.max_length:
                raw = raw[:self.max_length]
            if len(raw) < self.max_length:
                raw += [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * \
                    (self.max_length - len(raw) - 1)
            preds.append(raw)

        # [batch_size x max_length] (value: id)
        return torch.tensor(preds, dtype=torch.int64, device=device), None

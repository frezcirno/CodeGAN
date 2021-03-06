from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from beam import Beam
from .roberta import RobertaModel
from .decoder import TransformerDecoder
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


class CodeBert(nn.Module):
    """
        Generator model, robert + transformer

        Parameters:

        * `hidden_size` - size of hidden layer
        * `vocab_size` - size of vocab
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        beam_size: int,
        max_length: int,
    ):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.decoder = TransformerDecoder(hidden_size)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.max_length = max_length

        # register_buffer() - 模型的常量参数, 不会被训练
        # 类似:
        # [[0,   -1e4, -1e4],
        #  [0,    0,   -1e4],
        #  [0,    0,   0   ]]
        self.register_buffer("bias", torch.triu(torch.full((2048, 2048), -1e4), diagonal=1))

        # nn.Embedding(num_embeddings, embedding_dim) - Embedding层
        # 一个大小为num_embeddings的lookup-table，将一个词[0,num_embeddings) 转化为一个embedding_dim维的向量
        # 输入：[... x source_size] (value: [0,num_embeddings) )
        # 输出：[... x source_size x embedding_dim]

        # nn.Linear(in_features, out_features) - Linear层
        # 输入：[... x in_features]
        # 输出：[... x out_features]
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._tie_or_clone_weights(self.lm_head, self.encoder.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        output_embeddings.weight = input_embeddings.weight

    def encode(self, source_ids, source_mask):
        """
        Output:
            context: [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask: same as source_mask
        """
        # type: BaseModelOutputWithPoolingAndCrossAttentions
        context = self.encoder(source_ids, attention_mask=source_mask)

        memory_key_padding_mask = (1 - source_mask).bool()

        return context, memory_key_padding_mask

    def __decode(
        self,
        target_ids: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Input:
            target_ids: [batch_size x tgt_len]
            context: [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask: same as source_mask
        Output:
            decode_output: [batch_size x tgt_len x hidden_size]
        """
        # Embedding target
        tgt = self.encoder.embeddings(target_ids)  # [batch_size x tgt_len x hidden_size]
        tgt_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]  # [tgt_len x tgt_len]

        # Decode
        # [batch_size x tgt_len x hidden_size]
        decode_output = self.decoder(
            tgt=tgt,
            tgt_mask=tgt_mask,  # [tgt_len x tgt_len]
            memory=memory,  # [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return decode_output

    def decode(
        self,
        target_ids: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Input:
            target_ids: [batch_size x tgt_max_len]
            context: [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask: same as source_mask
        Output:
            logits: [batch_size x tgt_max_len x vocab_size]
        """
        decode_output = self.__decode(target_ids, memory, memory_key_padding_mask)
        out = torch.tanh(self.dense(decode_output))
        return self.lm_head(out)

    def decode_last(
        self,
        target_ids: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Input:
            target_ids: [batch_size x tgt_len]
            context: [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask: same as source_mask
        Output:
            last_logits: [batch_size x vocab_size]
        """
        decode_output = self.__decode(target_ids, memory, memory_key_padding_mask)
        last_output = decode_output[:, -1, :]
        return self.lm_head(self.dense(last_output).tanh())

    def get_memory_padding_mask(self, source_mask):
        return (1 - source_mask).bool()

    def forward(
        self,
        source_ids=None,
        source_mask=None,
        target_ids=None,
        target_mask=None,
        memory=None,
        memory_key_padding_mask=None,
        rewards=None,
        init_given_num=None,
        beam_search=False,
    ):
        """
        has target_mask -> get_loss
        init_given_num -> rollout_predict
        else -> greedy_predict
        """
        if init_given_num is not None:
            if memory is None:
                memory, memory_key_padding_mask = self.encode(source_ids, source_mask)
            return self.rollout_predict(memory, memory_key_padding_mask, target_ids, init_given_num)

        if rewards is not None:
            if memory is None:
                memory, memory_key_padding_mask = self.encode(source_ids, source_mask)
            return self.gan_get_loss(memory, memory_key_padding_mask, target_ids, rewards)

        if target_mask is not None:
            if memory is None:
                memory, memory_key_padding_mask = self.encode(source_ids, source_mask)
            return self.get_loss(memory, memory_key_padding_mask, target_ids, target_mask)

        if beam_search:
            return self.beam_predict(source_ids, source_mask)

        return self.predict(source_ids, source_mask)

    def get_loss(self, memory, memory_key_padding_mask, target_ids, target_mask):
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

        # [batch_size x src_max_len x args.hidden_size]
        logits = self.decode(target_ids, memory, memory_key_padding_mask)

        # Truncate
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

    def beam_predict(self, source_ids, source_mask, _device=None):
        """
        Inputs:
            source_ids: [batch_size x src_max_len]
            source_mask: [batch_size x src_max_len]

        Outputs:
            [batch_size x max_length] (value: id), padded by pad_token_id
        """
        context, _ = self.encode(source_ids, source_mask)

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
                lm_logits = self.decode_last(btarget_ids, ctx, memory_key_padding_mask)
                lm_logits = F.log_softmax(lm_logits, dim=1)  # [beam_size x vocab_size] (value: probs)

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

        memory, memory_key_padding_mask = self.encode(source_ids, source_mask)

        target_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.int64, device=device)

        # has_end = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(self.max_length - 1):
            # [batch_size x vocab_size] (value: probs)
            last_logits = self.decode_last(target_ids, memory, memory_key_padding_mask)

            # out = F.log_softmax(last_logits, dim=1)
            out = last_logits

            pred = out.argmax(1)  # [batch_size ] (values: id)
            target_ids = torch.cat([target_ids, pred.unsqueeze(1)], 1)

        # target_ids: [batch_size x max_length]
        # [batch_size x max_length x vocab_size]

        return mask_target(target_ids)

    def rollout_predict(
        self,
        memory, memory_key_padding_mask, init_target_ids, init_given_num
    ):
        """
        Predict like self.predict(), but
            1) sampling use multinomial()
            2) initial input is given rather than bos

        Inputs:
            context: [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            init_target_ids: [batch_size x init_length], generated target by self.predict()

        Outputs:
            target_ids: [batch_size x max_length]
            target_mask: [batch_size x max_length]
        """
        target_ids = init_target_ids[:, :init_given_num]

        for _ in range(init_given_num, self.max_length):
            # [batch_size x i]
            last_logit = self.decode_last(target_ids, memory, memory_key_padding_mask)
            last_logit = F.softmax(last_logit, dim=1)
            # [batch_size x vocab_size]
            pred = torch.multinomial(last_logit, 1)  # sampling one sample
            # [batch_size ]

            target_ids = torch.cat([target_ids, pred], 1)

        # target_ids: [batch_size x max_length]
        return mask_target(target_ids)

    def gan_get_loss(self, memory, memory_key_padding_mask, target_ids, rewards):
        """
        Inputs:
            memory: [batch_size x src_max_len x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            target_ids: [batch_size x tgt_max_len], generated target by self.predict()
            rewards: [batch_size x src_max_len]

        Outputs:
            loss: (sum of a batch)
        """
        logits = self.decode(target_ids, memory, memory_key_padding_mask)  # [batch_size x tgt_max_len x vocab_size]
        out = F.log_softmax(logits, dim=2)
        shift_logits = out[:, :-1, :]
        # [batch_size x tgt_max_len-1 x vocab_size]

        flat_lm_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        # [batch_size*tgt_max_len-1, vocab_size]
        flat_target_ids: Tensor = target_ids[..., 1:].reshape(-1)
        flat_rewards: Tensor = rewards[..., 1:].reshape(-1)
        # [batch_size*tgt_max_len-1]

        """
        shift_logits[i]: the next word vocab
        flat_target_ids[i]: the real next word
        flat_rewards[i]: the reward of using the word
        """
        flat_target_ids *= ~flat_target_ids.eq(tokenizer.pad_token_id)
        flat_target_onehot: Tensor = F.one_hot(flat_target_ids, self.vocab_size).float()

        # [batch_size*tgt_max_len-1]
        chosen_logits = torch.sum(flat_lm_logits * flat_target_onehot, dim=1)
        loss = torch.sum(chosen_logits * flat_rewards)

        # loss = torch.tensor(0.0, device=flat_lm_logits.device)
        # for i, vocab in enumerate(flat_lm_logits):
        #     choice = flat_target_ids[i]
        #     if choice != tokenizer.pad_token_id:
        #         loss += vocab[choice] * flat_rewards[i]

        return -loss

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .roberta import RobertaModel
from .decoder import TransformerDecoder
from ... import tokenize


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


class Generator(nn.Module):
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
        super(Generator, self).__init__()
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

        self.lsm = nn.LogSoftmax(dim=-1)

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
            lm_logits: [batch_size x tgt_max_len x vocab_size]
        """
        decode_output = self.__decode(target_ids, memory, memory_key_padding_mask)
        return self.lm_head(self.dense(decode_output).tanh())

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
        source_ids,
        source_mask,
        target_ids=None,
        target_mask=None,
        rewards=None,
        rollout=False,
        init_given_num=None,
        beam_search=False,
    ):
        """
        has target_mask -> get_loss
        rollout=True -> rollout_predict
        else -> greedy_predict
        """
        if rollout:
            return self.rollout_predict(source_ids, source_mask, target_ids, init_given_num)

        if rewards is not None:
            return self.gan_get_loss(source_ids, source_mask, target_ids, rewards)

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

        # [batch_size x src_max_len x args.hidden_size]
        memory, memory_key_padding_mask = self.encode(source_ids, source_mask)

        lm_logits = self.decode(target_ids, memory, memory_key_padding_mask)

        # Truncate
        # [batch_size x (tgt_max_len-1) x vocab_size]
        shift_logits = lm_logits[:, :-1, :]

        # Eval bos_token is invalid
        # [(batch_size*(tgt_max_len-1)) ]
        active_target = target_ids[..., 1:].reshape(-1)
        active_loss = target_mask[..., 1:].ne(0).reshape(-1)

        # Flatten the tokens
        loss = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size)[active_loss],
            # [batch_size*(tgt_max_len-1) x vocab_size]
            active_target[active_loss],  # [batch_size*(tgt_max_len-1) ]  (remove bos_token)
            ignore_index=tokenize.pad_token_id,
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
                lm_logits = self.decode_last(btarget_ids, ctx, memory_key_padding_mask)  # [beam_size x hidden_size]

                # [beam_size x vocab_size] (value: probs)
                out = self.lsm(lm_logits).data

                beam.advance(out)

                # generate a word
                # copy_() - similar to assign
                btarget_ids.data.copy_(
                    # index_select() - choose row(0) by indexes
                    btarget_ids.data.index_select(0, beam.getCurrentOrigin())
                )
                btarget_ids = torch.cat(
                    (btarget_ids, beam.getCurrentState(source_ids.device)), -1
                )
                # [beam_size x i]

            # [beam_size x [n]] (values: token_id)
            hyp = beam.getHyp(beam.getFinal())
            # truncate
            beam_preds = beam.buildTargetTokens(hyp)[: self.beam_size]
            # [beam_size x <=max_length]

            best = beam_preds[0]
            raw = [tokenize.bos_token_id] + [x.item() for x in best]
            if len(raw) > self.max_length:
                raw = raw[:self.max_length]
            if len(raw) < self.max_length:
                raw += [tokenize.eos_token_id] + [tokenize.pad_token_id] * \
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

        target_ids = torch.full((batch_size, 1), tokenize.bos_token_id, dtype=torch.int64, device=device)

        # has_end = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(self.max_length - 1):
            # [batch_size x vocab_size] (value: probs)
            last_logits = self.decode_last(target_ids, memory, memory_key_padding_mask)

            # out: Tensor = self.lsm(last_logits)
            out = last_logits

            pred = out.argmax(1)  # [batch_size ] (values: id)
            target_ids = torch.cat([target_ids, pred.unsqueeze(1)], 1)

        # target_ids: [batch_size x max_length]
        # [batch_size x max_length x vocab_size]

        return mask_target(target_ids)

    def rollout_predict(
        self, context, memory_key_padding_mask, init_target_ids, init_given_num
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
            last_logit = self.decode_last(target_ids, context, memory_key_padding_mask)
            # [batch_size x vocab_size]

            pred = torch.multinomial(last_logit.softmax(1), 1)  # sampling one sample
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
        lm_logits = self.decode(target_ids, memory, memory_key_padding_mask)  # [batch_size x tgt_max_len x vocab_size]
        shift_logits = lm_logits[:, :-1, :]
        # [batch_size x tgt_max_len-1 x vocab_size]

        flat_lm_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        # [batch_size*tgt_max_len-1 x vocab_size]
        flat_target_ids = target_ids[..., 1:].reshape(-1)
        flat_rewards = rewards[..., 1:].reshape(-1)
        # [batch_size*tgt_max_len-1]

        """
        shift_logits[i]: the next word vocab
        flat_target_ids[i]: the real next word
        flat_rewards[i]: the reward of using the word
        """
        loss = torch.tensor(0.0, device=flat_lm_logits.device)
        for i, vocab in enumerate(flat_lm_logits):
            choice = flat_target_ids[i]
            if choice != tokenize.pad_token_id:
                loss += vocab[choice] * flat_rewards[i]

        return -loss


class Beam(object):
    def __init__(self, size, device):
        self.size = size
        # The score for each translation on the beam.
        self.scores = torch.zeros(size, dtype=torch.float, device=device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.zeros(size, dtype=torch.long, device=device)]
        self.nextYs[0][0] = tokenize.bos_token_id
        # Has EOS topped the beam yet.
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self, device):
        """
        Get the outputs for the current timestep.
        Return: [beam_size x 1]
        """
        return self.nextYs[-1].clone().detach().view(-1, 1).to(device)

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step [beam_size x vocab_size]
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == tokenize.eos_token_id:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode="trunc")
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == tokenize.eos_token_id:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == tokenize.eos_token_id:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != tokenize.eos_token_id:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[: self.size - len(self.finished)]
        return self.finished[: self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == tokenize.eos_token_id:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

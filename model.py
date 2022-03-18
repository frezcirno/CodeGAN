import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tokenizer
# import pytorch_lightning as pl

# from transformers import RobertaModel
from modeling_roberta import RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


def get_target_mask(target_ids: Tensor):
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
        device,
    ):
        super(Generator, self).__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=12), num_layers=6,
        )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.max_length = max_length
        self.device = device

        # register_buffer() - 模型的常量参数, 不会被训练
        # 类似:
        # [[0,   -1e4, -1e4],
        #  [0,    0,   -1e4],
        #  [0,    0,   0   ]]
        self.register_buffer(
            "bias", torch.triu(torch.full((2048, 2048), -1e4), diagonal=1)
        )

        self.dense = nn.Linear(hidden_size, hidden_size)

        # nn.Embedding(num_embeddings, embedding_dim) - Embedding层
        # 一个大小为num_embeddings的lookup-table，将一个词[0,num_embeddings) 转化为一个embedding_dim维的向量
        # 输入：[... x source_size] (value: [0,num_embeddings) )
        # 输出：[... x source_size x embedding_dim]

        # nn.Linear(in_features, out_features) - Linear层
        # 输入：[... x in_features]
        # 输出：[... x out_features]

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_weights(self.lm_head, self.encoder.get_input_embeddings())

        self.lsm = nn.LogSoftmax(dim=-1)

    def tie_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        first_module.weight = second_module.weight

    def get_context(self, source_ids, source_mask):
        """
        Output:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask: same as source_mask
        """
        # type: BaseModelOutputWithPoolingAndCrossAttentions
        context = self.encoder(source_ids, attention_mask=source_mask)

        memory_key_padding_mask = (1 - source_mask).bool()

        return context["last_hidden_state"], memory_key_padding_mask

    def forward(
        self,
        context,
        memory_key_padding_mask,
        target_ids=None,
        target_mask=None,
        rewards=None,
        rollout=False,
        init_given_num=None,
    ):
        """
        has target_mask -> get_loss
        rollout=True -> rollout_predict
        else -> greedy_predict
        """
        if rollout:
            return self.rollout_predict(
                context, memory_key_padding_mask, target_ids, init_given_num
            )

        if rewards is not None:
            return self.gan_get_loss(
                context, memory_key_padding_mask, target_ids, rewards
            )

        if target_mask is not None:
            return self.get_loss(
                context, memory_key_padding_mask, target_ids, target_mask
            )

        return self.predict(context, memory_key_padding_mask)

    def get_loss(self, context, memory_key_padding_mask, target_ids, target_mask):
        """
        Inputs:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            target_ids: [batch_size x target_length]
                        values: [id(bos), ..., id(eos), n * id(pad)]
            target_mask: [batch_size x target_length]
                        values: [   1,    ...,    1,    n * 0]

        Outputs:
            loss: average of all loss
            active_loss: the number of loss
            lm_logits: [batch_size x target_length x vocab_size]
        """

        # attn_mask: [target_length x target_length]
        attn_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]

        # Embedding target
        tgt_embeddings = (
            # [...target_ids x hidden_size]
            self.encoder.embeddings(target_ids)
            .permute([1, 0, 2])
            .contiguous()
        )  # [target_length x batch_size x hidden_size]

        # Decode
        out = self.decoder(
            tgt_embeddings,  # [target_length x batch_size x hidden_size]
            # [batch_size x source_length x hidden_size]
            context.permute([1, 0, 2]),
            tgt_mask=attn_mask,  # [target_length x target_length]
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [target_length x batch_size x hidden_size]

        # Activate
        # [batch_size x target_length x hidden_size], starts with the first valid token
        hidden_states = self.dense(out).tanh().permute([1, 0, 2]).contiguous()
        lm_logits = self.lm_head(hidden_states)

        # Truncate
        # [batch_size x (target_length-1) x vocab_size]
        shift_logits = lm_logits[:, :-1, :].contiguous()

        # Eval bos_token is invalid
        # [(batch_size*(target_length-1)) ]
        active_target = target_ids[..., 1:].reshape(-1)
        active_loss = target_mask[..., 1:].ne(0).reshape(-1)

        # Flatten the tokens
        loss = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size)[active_loss],
            # [batch_size*(target_length-1) x vocab_size]
            active_target[active_loss],
            # [batch_size*(target_length-1) ]  (remove bos_token)
            ignore_index=tokenizer.pad_token_id,
        )

        # return loss, loss * active_loss.sum(), active_loss.sum()
        return loss, active_loss.sum(), lm_logits

    def beam_predict(self, source_ids, source_mask, _device=None):
        """
        Inputs:
            source_ids: [batch_size x source_length]
            source_mask: [batch_size x source_length]

        Outputs:
            [batch_size x max_length] (value: id), padded by pad_token_id
        """
        context, _ = self.get_context(source_ids, source_mask)
        context = context.permute([1, 0, 2]).contiguous()

        preds = []
        batch_size = source_ids.size(0)
        device = _device if _device else source_ids.device

        # Beam search for every sample
        for i in range(batch_size):
            beam = Beam(self.beam_size, source_ids.device)
            btarget_ids = beam.getCurrentState(source_ids.device)
            # [beam_size x 1]
            ctx = context[:, i: i + 1].repeat(1, self.beam_size, 1)
            # [source_length x beam_size x hidden_size]
            context_mask = source_mask[i: i + 1, :].repeat(self.beam_size, 1)
            memory_key_padding_mask = (1 - context_mask).bool()
            # [beam_size x source_length]
            for _ in range(self.max_length):
                if beam.done():
                    break
                attn_mask = self.bias[: btarget_ids.shape[1],
                                      : btarget_ids.shape[1]]
                tgt_embeddings = (
                    self.encoder.embeddings(btarget_ids)
                    # [beam_size x i x hidden_size]
                    .permute([1, 0, 2]).contiguous()
                )
                out = self.decoder(
                    tgt_embeddings,  # [i x beam_size x hidden_size]
                    ctx,  # [source_length x beam_size x hidden_size]
                    tgt_mask=attn_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )  # [i x beam_size x hidden_size]
                out = self.dense(out).tanh()
                hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                # [beam_size x hidden_size]
                out = self.lsm(self.lm_head(hidden_states)).data
                # [beam_size x vocab_size] (value: probs)
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
            raw = [tokenizer.bos_token_id] + [x.item() for x in best]
            if len(raw) > self.max_length:
                raw = raw[:self.max_length]
            if len(raw) < self.max_length:
                raw += [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * \
                    (self.max_length - len(raw) - 1)
            preds.append(raw)

        # [batch_size x max_length] (value: id)
        return torch.tensor(preds, dtype=torch.int64, device=device)

    def predict(self, context, memory_key_padding_mask):
        """
        Predict use greedy search
        <s> -> a
        <s>,a -> a,b
        <s>,a,b -> a,b,c
        ...
        <s>,...,y -> ...,y,z
        return a,...,z

        Inputs:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()

        Outputs:
            target_ids: [batch_size x max_length], with bos
            target_mask: [batch_size x max_length]
        """
        batch_size = context.size(0)

        target_ids = torch.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=torch.int64, device=context.device
        )

        # has_end = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(self.max_length - 1):
            # [batch_size x i]
            attn_mask = self.bias[: target_ids.size(1), : target_ids.size(1)]
            tgt_embeddings = (
                self.encoder.embeddings(target_ids)  # [... x hidden_size]
                .permute([1, 0, 2])
                .contiguous()
            )  # [i x batch_size x hidden_size]
            out = self.decoder(
                tgt_embeddings,
                # [i x batch_size x hidden_size]
                context.permute([1, 0, 2]),
                # [source_length x batch_size x hidden_size]
                tgt_mask=attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # [i x batch_size x hidden_size]
            out = self.dense(out).tanh()
            hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
            # [batch_size x hidden_size]
            out = self.lsm(self.lm_head(hidden_states))
            # [batch_size x vocab_size] (value: probs)

            pred = out.argmax(1)  # [batch_size ] (values: id)
            target_ids = torch.cat([target_ids, pred.unsqueeze(1)], 1)

        # target_ids: [batch_size x max_length]
        # [batch_size x max_length x vocab_size]

        return get_target_mask(target_ids)

    def rollout_predict(
        self, context, memory_key_padding_mask, init_target_ids, init_given_num
    ):
        """
        Predict like self.predict(), but
            1) sampling use multinomial()
            2) initial input is given rather than bos

        Inputs:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            init_target_ids: [batch_size x init_length], generated target by self.predict()

        Outputs:
            target_ids: [batch_size x max_length]
            target_mask: [batch_size x max_length]
        """
        target_ids = init_target_ids[:, :init_given_num]

        for _ in range(init_given_num, self.max_length):
            # [batch_size x i]
            tgt_embeddings = (
                self.encoder.embeddings(target_ids).permute(
                    [1, 0, 2]).contiguous()
            )  # [i x batch_size x hidden_size]

            attn_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]

            out = self.decoder(
                tgt_embeddings,  # [i x batch_size x hidden_size]
                context.permute([1, 0, 2]),
                # [source_length x batch_size x hidden_size]
                tgt_mask=attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # [i x batch_size x hidden_size]
            out = out[-1, ...]
            out = self.dense(out)
            out = self.lm_head(out)
            # [batch_size x vocab_size]

            pred = torch.multinomial(out.softmax(1), 1)  # sampling one sample
            # [batch_size ]

            target_ids = torch.cat([target_ids, pred], 1)

        # target_ids: [batch_size x max_length]
        return get_target_mask(target_ids)

    def gan_get_loss(self, context, memory_key_padding_mask, target_ids, rewards):
        """
        Inputs:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            target_ids: [batch_size x target_length], generated target by self.predict()
            rewards: [batch_size x source_length]

        Outputs:
            loss: (sum of a batch)
        """
        tgt_embeddings = (
            self.encoder.embeddings(target_ids)  # [... x hidden_size]
            .permute([1, 0, 2])
            .contiguous()
        )  # [target_length x batch_size x hidden_size]

        attn_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]

        out = self.decoder(
            tgt_embeddings,  # [target_length x batch_size x hidden_size]
            # [source_length x batch_size x hidden_size]
            context.permute([1, 0, 2]),
            tgt_mask=attn_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        # <s>,a,b...,<e> -> a,b...,<e>,<?>
        out = out.permute([1, 0, 2]).contiguous()
        out = self.dense(out).tanh()
        out = self.lm_head(out)
        lm_logits = self.lsm(out)
        # [batch_size x target_length x vocab_size]
        shift_logits = lm_logits[:, :-1, :]
        # [batch_size x target_length-1 x vocab_size]

        flat_lm_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        # [batch_size*target_length-1 x vocab_size]
        flat_target_ids = target_ids[..., 1:].reshape(-1)
        flat_rewards = rewards[..., 1:].reshape(-1)
        # [batch_size*target_length-1]

        """
        shift_logits[i]: the next word vocab
        flat_target_ids[i]: the real next word
        flat_rewards[i]: the reward of using the word
        """
        loss = torch.tensor(0.0, device=flat_lm_logits.device)
        for i, vocab in enumerate(flat_lm_logits):
            choice = flat_target_ids[i]
            if choice != tokenizer.pad_token_id:
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
        self.nextYs[0][0] = tokenizer.bos_token_id
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
                if self.nextYs[-1][i] == tokenizer.eos_token_id:
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
            if self.nextYs[-1][i] == tokenizer.eos_token_id:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == tokenizer.eos_token_id:
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
                if self.nextYs[-1][i] != tokenizer.eos_token_id:
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
                if tok == tokenizer.eos_token_id:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels,
                  kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        nn.init.constant_(m.bias.data, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


class Highway(nn.Module):
    def __init__(
        self,
        input_size,
    ):
        super(Highway, self).__init__()

        self.normal = Linear(input_size, input_size)
        self.gate = Linear(input_size, input_size)

    def forward(self, x):
        y = F.relu(self.normal(x))
        t = torch.sigmoid(self.gate(x))

        return y * t + (1 - t) * x


class Conv2dHub(nn.Module):
    def __init__(self, filter_sizes, filter_nums, hidden_size, max_length):
        super(Conv2dHub, self).__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(
                # [bat_siz x chan(1) x max_len x hid_siz]
                Conv2d(1, filter_num, (filter_size, hidden_size)),
                # [bat_siz x fil_num x (max_len-fil_siz+1) x 1]
                nn.BatchNorm2d(filter_num),
                nn.ReLU(),
                nn.MaxPool2d((max_length - filter_size + 1, 1)),
                # [bat_siz x fil_num x 1 x 1]
            )
            for filter_size, filter_num in zip(filter_sizes, filter_nums)
        ])

    def forward(self, i):
        out = []
        for conv2d in self.module_list:
            # [bat_siz x fil_num x 1 x 1]
            x = conv2d(i).squeeze(-1).squeeze(-1)
            out.append(x)

        # [bat_siz x fil_sum]
        out = torch.cat(out, dim=1)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        source_length,
        target_length,
        vocab_size,
        hidden_size,
        device,
        max_filter_size=32,
    ):
        super(Discriminator, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device

        self.src_embedding = Embedding(
            vocab_size, hidden_size, tokenizer.pad_token_id)
        self.tgt_embedding = Embedding(
            vocab_size, hidden_size, tokenizer.pad_token_id)

        filter_sizes = list(range(1, max_filter_size, 4))
        filter_nums = [100 + i * 10 for i in range(1, max_filter_size, 4)]
        filter_sum = sum(filter_nums)
        # filter_sizes: [1, 5, 9, 13, 17, ...]
        # filter_nums: [110, 150, 190, ...]

        self.src_filters = Conv2dHub(
            filter_sizes, filter_nums, hidden_size, source_length)
        self.tgt_filters = Conv2dHub(
            filter_sizes, filter_nums, hidden_size, target_length)

        self.src_highway = Highway(filter_sum)
        self.tgt_highway = Highway(filter_sum)

        # self.src_drop = nn.Dropout()
        # self.tgt_drop = nn.Dropout()

        self.hidden2out = Linear(2 * filter_sum, 1)

    def forward(self, source_ids: Tensor, target_ids: Tensor):
        """
            Inputs:
            source_ids: [batch_size x source_length] (value: [0, vocab_size))
            target_ids: [batch_size x target_length] (value: [0, vocab_size))

            Outputs:
            scores: batch_size (value: 0 to 1)
        """
        x = self.src_embedding(source_ids).unsqueeze(1)
        # [batch_size x chan(1) x source_length x hidden_size]
        y = self.tgt_embedding(target_ids).unsqueeze(1)
        # [batch_size x chan(1) x target_length x hidden_size]

        src_outputs = self.src_filters(x)
        tgt_outputs = self.tgt_filters(y)
        # [batch_size x filter_sum]

        src_outputs = self.src_highway(src_outputs)
        tgt_outputs = self.tgt_highway(tgt_outputs)

        # src_outputs = self.src_drop(src_outputs)
        # tgt_outputs = self.tgt_drop(tgt_outputs)
        # [batch_size x filter_sum]

        src_tgt = torch.cat([src_outputs, tgt_outputs], dim=1)
        # [batch_size x filter_sum*2]

        # [batch_size], Get scores [0-1]
        logits = self.hidden2out(src_tgt).squeeze(1)
        scores = torch.sigmoid(logits)
        return scores

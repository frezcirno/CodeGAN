import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl

# from transformers import RobertaModel
from modeling_roberta import RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class Generator(nn.Module):
    """
        Generator model, robert + transformer

        Parameters:

        * `hidden_size` - size of hidden layer
        * `vocab_size` - size of vocab
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `bos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
        * `pad_id`- padding ids.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        beam_size: int,
        max_length: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
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
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device_ids = [device]

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
                        values: [id(cls), ..., id(sep), n * id(pad)]
            target_mask: [batch_size x target_length]
                        values: [   1,    ...,    1,    n * 0]

        Outputs:
            loss
            active_loss_sum
            lm_logits: [batch_size x target_length x vocab_size]
        """

        # attn_mask: [target_length x target_length]
        attn_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]

        # Embedding target
        tgt_embeddings = (
            self.encoder.embeddings(target_ids)  # [...target_ids x hidden_size]
            .permute([1, 0, 2])
            .contiguous()
        )  # [target_length x batch_size x hidden_size]

        # Decode
        out = self.decoder(
            tgt_embeddings,  # [target_length x batch_size x hidden_size]
            context.permute([1, 0, 2]),  # [batch_size x source_length x hidden_size]
            tgt_mask=attn_mask,  # [target_length x target_length]
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [target_length x batch_size x hidden_size]

        # Activate
        hidden_states = self.dense(out).tanh().permute([1, 0, 2]).contiguous()
        # [batch_size x target_length x hidden_size]

        lm_logits = self.lm_head(hidden_states)
        # [batch_size x target_length x vocab_size]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        # Remove eos_token
        # [batch_size x (target_length-1) x vocab_size]
        active_loss = target_mask[..., 1:].ne(0).reshape(-1)
        # Remove bos_token
        # [(batch_size*(target_length-1)) ]

        # Flatten the tokens
        loss = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size)[active_loss],
            # [batch_size*(target_length-1) x vocab_size]
            target_ids[..., 1:].reshape(-1)[active_loss],
            # [batch_size*(target_length-1) ]  (remove bos_token)
            ignore_index=-1,
        )

        # return loss, loss * active_loss.sum(), active_loss.sum()
        return loss, active_loss.sum(), lm_logits

    def beam_predict(self, source_ids, source_mask):
        """
        Inputs:
            source_ids: [batch_size x source_length]
            source_mask: [batch_size x source_length]

        Outputs:
            [batch_size x max_length] (value: id), padded by self.pad_id
        """
        context, _ = self.get_context(source_ids, source_mask)
        context = context.permute([1, 0, 2]).contiguous()

        preds = []
        batch_size = source_ids.size(0)

        # Beam search for every sample
        for i in range(batch_size):
            beam = Beam(self.beam_size, self.bos_id, self.eos_id)
            btarget_ids = beam.getCurrentState(source_ids.device)
            # [beam_size x 1]
            ctx = context[:, i : i + 1].repeat(1, self.beam_size, 1)
            # [source_length x beam_size x hidden_size]
            context_mask = source_mask[i : i + 1, :].repeat(self.beam_size, 1)
            memory_key_padding_mask = (1 - context_mask).bool()
            # [beam_size x source_length]
            for _ in range(self.max_length):
                if beam.done():
                    break
                attn_mask = self.bias[: btarget_ids.shape[1], : btarget_ids.shape[1]]
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
                    (btarget_ids, beam.getCurrentState(btarget_ids.device)), -1
                )
                # [beam_size x i]

            hyp = beam.getHyp(beam.getFinal())  # [beam_size x [n]] (values: token_id)
            # truncate
            beam_preds = beam.buildTargetTokens(hyp)[: self.beam_size]
            # [beam_size x <=max_length]

            best = beam_preds[0]
            raw = [self.bos_id] + [x.item() for x in best]
            if len(raw) < self.max_length:
                raw += [self.eos_id] + [self.pad_id] * (self.max_length - len(raw) - 1)
            preds.append(raw)

        # [batch_size x max_length] (value: id)
        return torch.tensor(preds, dtype=torch.int64, device=source_ids.device)

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
            preds: [batch_size x max_length], with bos
            hiddens: [batch_size x max_length x vocab_size] (value: probs)
                the hidden outputs every step
        """
        batch_size = context.size(0)

        target_ids = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.int64, device=context.device
        )

        hiddens = []
        # has_end = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(self.max_length - 1):
            # [batch_size x i]
            attn_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]
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

            hiddens.append(out)
            pred = out.argmax(1)  # [batch_size ] (values: id)
            target_ids = torch.cat([target_ids, pred.unsqueeze(1)], 1)

            # has_end = has_end.logical_or(pred.cpu() == self.eos_id)
            # if all(has_end):
            #     break

        # Padding to self.max_length, filling zeros
        # if target_ids.size(1) < self.max_length:
        #     padding = torch.zeros(
        #         batch_size,
        #         self.max_length - target_ids.size(1),
        #         device=target_ids.device,
        #     )
        #     target_ids = torch.cat([target_ids, padding], 1)

        # target_ids: [batch_size x max_length]
        hiddens = torch.stack(hiddens, dim=1)  # [batch_size x max_length x vocab_size]

        return target_ids, hiddens

    def rollout_predict(
        self, context, memory_key_padding_mask, init_target_ids, init_given_num
    ):
        """
        Predict like self.predict(), but
            1) sampling use multinomial()
            2) initial input is given rather than eos

        Inputs:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            init_target_ids: [batch_size x init_length], generated target by self.predict()

        Outputs:
             : [batch_size x max_length]
        """
        target_ids = init_target_ids[:, :init_given_num]

        for _ in range(init_given_num, self.max_length):
            # [batch_size x i]
            tgt_embeddings = (
                self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
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

            ## here: exp() or log()?
            pred = torch.multinomial(out.exp(), 1)  # sampling one sample
            # [batch_size ]

            target_ids = torch.cat([target_ids, pred], 1)

        # target_ids: [batch_size x max_length]
        return target_ids

    def gan_get_loss(self, context, memory_key_padding_mask, target_ids, rewards):
        """
        Inputs:
            context: [batch_size x source_length x hidden_size]
            memory_key_padding_mask = (1 - source_mask).bool()
            target_ids: [batch_size x source_length], generated target by self.predict()
            target_mask: [batch_size x source_length]
            rewards: [batch_size x source_length]

        Outputs:
             : [batch_size x max_length]
        """
        tgt_embeddings = (
            self.encoder.embeddings(target_ids)  # [... x hidden_size]
            .permute([1, 0, 2])
            .contiguous()
        )  # [target_length x batch_size x hidden_size]

        attn_mask = self.bias[: target_ids.shape[1], : target_ids.shape[1]]

        out = self.decoder(
            tgt_embeddings,  # [target_length x batch_size x hidden_size]
            context.permute([1, 0, 2]),  # [source_length x batch_size x hidden_size]
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
            if choice != self.pad_id:
                loss += vocab[choice] * flat_rewards[i]

        return -loss


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        # The score for each translation on the beam.
        self.scores = torch.cuda.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.cuda.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
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
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode='trunc')
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
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
                if self.nextYs[-1][i] != self._eos:
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
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class Rollout:
    def __init__(self, gen, dis, max_length):
        self.gen = gen
        self.dis = dis
        self.max_length = max_length
        self.gen_device = gen.device_ids[0]
        self.dis_device = dis.device_ids[0]

    def get_reward(
        self,
        source_ids,
        context,
        memory_key_padding_mask,
        pre_target_ids,
        pre_target_mask,
        rollnum=20,
    ):
        """
        Rollout method:
            give none(<s>), predict, get reward
            give pre_target_ids[:1](<s>,a), predict, get reward
            give pre_target_ids[:2](<s>,a,b), predict, get reward
            ...
            give pre_target_ids[:max_length-1], predict, get reward
        Input:
            pre_target_ids: [batch_size x target_length], with bos!!
        Outputs:
            rewards: [batch_size x max_length]
                rewards[i][0] is empty
                rewards[i][j]: the reward of using word Seq[j] as the next of Seq[0..j-1].
        """
        batch_size = context.size(0)

        rewards = torch.zeros(
            self.max_length, batch_size, dtype=torch.float, device=self.dis_device
        )
        self.dis.eval()
        for _ in range(rollnum):  # rollout times, mean() later
            # ignore bos_token
            for init_given_num in range(2, self.max_length):
                if not any(pre_target_mask[:, init_given_num]):
                    break
                target_ids = self.gen(
                    context,
                    memory_key_padding_mask,
                    target_ids=pre_target_ids,
                    rollout=True,
                    init_given_num=init_given_num,
                )
                with torch.no_grad():
                    pred = self.dis(
                        source_ids.to(self.dis_device), target_ids.to(self.dis_device)
                    )
                # pred = pred.cpu()
                rewards[init_given_num - 1] += pred

            with torch.no_grad():
                pred = self.dis(
                    source_ids.to(self.dis_device), pre_target_ids.to(self.dis_device)
                )
            # pred = pred.cpu()
            # [batch_size]
            rewards[self.max_length - 1] += pred

        rewards = rewards.permute([1, 0]).contiguous()
        # rewards: [batch_size x max_length]

        rewards = rewards.to(pre_target_mask.device)
        rewards = rewards * pre_target_mask
        rewards = rewards / (1.0 * rollnum)
        return rewards


class Discriminator(nn.Module):
    def __init__(
        self,
        source_length,
        target_length,
        vocab_size,
        hidden_size,
        bos_token_id,
        eos_token_id,
        device,
    ):
        super(Discriminator, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.device_ids = [device]

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        src_filter_sizes = list(range(1, 32, 4))
        src_filter_nums = [(i * 10) for i in src_filter_sizes]
        # src_filter_sizes: [1, 5, 9, 13, 17, ...]
        # src_filter_nums: [110, 150, 190, ...]

        src_filter_sum = sum(src_filter_nums)

        self.src_filters = nn.ModuleList()

        for filter_size, filter_num in zip(src_filter_sizes, src_filter_nums):
            self.src_filters.append(
                nn.Sequential(
                    nn.Conv2d(1, filter_num, kernel_size=(filter_size, hidden_size)),
                    nn.BatchNorm2d(filter_num),
                    nn.ReLU(),
                    nn.MaxPool2d((source_length - filter_size + 1, 1), stride=None),
                )
            )

        self.src_output_lin = nn.Linear(src_filter_sum, src_filter_sum, bias=False)
        self.src_trans_lin = nn.Linear(src_filter_sum, src_filter_sum, bias=False)
        self.src_drop = nn.Dropout()

        tgt_filter_sizes = list(range(1, 32, 4))
        tgt_filter_nums = [(i * 10) for i in tgt_filter_sizes]
        # src_filter_sizes: [1, 5, 9, 13, 17, ...]
        # src_filter_nums: [110, 150, 190, ...]

        tgt_filter_sum = sum(tgt_filter_nums)

        self.tgt_filters = nn.ModuleList()

        for filter_size, filter_num in zip(tgt_filter_sizes, tgt_filter_nums):
            self.tgt_filters.append(
                nn.Sequential(
                    nn.Conv2d(1, filter_num, kernel_size=(filter_size, hidden_size)),
                    nn.BatchNorm2d(filter_num),
                    nn.ReLU(),
                    nn.MaxPool2d((target_length - filter_size + 1, 1), stride=None),
                )
            )

        self.tgt_output_lin = nn.Linear(tgt_filter_sum, tgt_filter_sum, bias=False)
        self.tgt_trans_lin = nn.Linear(tgt_filter_sum, tgt_filter_sum, bias=False)
        self.tgt_drop = nn.Dropout()

        self.hidden2out = nn.Linear(tgt_filter_sum + src_filter_sum, 1)

    def forward(self, source_ids: Tensor, target_ids: Tensor):
        """
	    Inputs:
            source_ids: [batch_size x source_length] (value: [0, vocab_size))
            target_ids: [batch_size x target_length] (value: [0, vocab_size))

	    Outputs:
            scores: batch_size (value: 0 to 1)
        """
        x = self.embedding(source_ids).unsqueeze(1)
        # [batch_size x chan(1) x source_length x hidden_size]

        src_outputs = []
        for src_filter in self.src_filters:
            x1 = src_filter(x)
            # batch_size x num_filter_i x 1 x 1
            x1 = x1.squeeze(-1).squeeze(-1)
            # batch_size x num_filter_i
            src_outputs.append(x1)
        src_outputs = torch.concat(src_outputs, dim=1)
        # [batch_size x num_filter_sum]

        output = self.src_output_lin(src_outputs).relu()
        t_gate = self.src_trans_lin(src_outputs).sigmoid()
        src_output = t_gate * output + (1.0 - t_gate) * src_outputs
        src_drop = self.src_drop(src_output)
        # [batch_size x num_filter_sum]

        y = self.embedding(target_ids).unsqueeze(1)
        # [batch_size x chan(1) x target_length x hidden_size]

        tgt_outputs = []
        for tgt_filter in self.tgt_filters:
            y1 = tgt_filter(y)
            # batch_size x num_filter_i x 1 x 1
            y1 = y1.squeeze(-1).squeeze(-1)
            # batch_size x num_filter_i
            tgt_outputs.append(y1)

        tgt_outputs = torch.concat(tgt_outputs, dim=1)
        # [batch_size x num_filter_sum]

        output = self.tgt_output_lin(tgt_outputs).relu()
        t_gate = self.tgt_trans_lin(tgt_outputs).sigmoid()
        tgt_output = t_gate * output + (1.0 - t_gate) * tgt_outputs
        tgt_drop = self.tgt_drop(tgt_output)
        # [batch_size x num_filter_sum]

        tgt_src = torch.concat([tgt_drop, src_drop], dim=1)
        # [batch_size x num_filter_sum*2]

        # Get scores [0-1]
        logits = self.hidden2out(tgt_src).squeeze(1)
        # [batch_size]
        scores = logits.sigmoid()
        return scores

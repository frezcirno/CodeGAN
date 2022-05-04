from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tokenizer


class Beam(object):
    def __init__(self, beam_size, device):
        self.beam_size = beam_size
        # The backpointers at each time-step.
        self.prevKs = []
        # The score for each translation on the beam.
        self.scores: Tensor
        # The outputs at each time-step. [1, beam_size]
        self.nextYs = [torch.zeros(beam_size, dtype=torch.long, device=device)]
        self.nextYs[0][0] = tokenizer.bos_token_id
        # Has EOS topped the beam yet.
        self.eosTop = False
        # Time and k pair for finished. List[Tuple[]]
        self.finished: List[Tuple[Tensor, int, int]] = []

    def getCurrentState(self, device):
        """
        Get the outputs for the current timestep.
        Return: [beam_size x 1]
        """
        return self.nextYs[-1].unsqueeze(1).to(device)

    def getCurrentOrigin(self) -> Tensor:
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, lsm_logits: Tensor):
        """
        Given prob over words for every last beam `wordLk`

        Parameters:

        * `lsm_logits`- probs of advancing from the last step [beam_size x vocab_size]
        """
        vocab_size = lsm_logits.size(1)

        # Sum the previous scores.
        if self.prevKs:
            beamLk = lsm_logits + self.scores.unsqueeze(1).expand_as(lsm_logits)  # [beam_size, vocab_size]

            # Don't let EOS have children.
            nextY = self.nextYs[-1]
            for i in range(nextY.size(0)):
                if nextY[i] == tokenizer.eos_token_id:
                    beamLk[i] = -1e20
        else:
            beamLk = lsm_logits[0]  # [vocab_size]

        flatBeamLk = beamLk.view(-1)  # [n * vocab_size]
        bestScores, bestScoresId = flatBeamLk.topk(self.beam_size, 0)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId.div(vocab_size, rounding_mode="trunc")
        nextY = bestScoresId - prevK * vocab_size
        self.prevKs.append(prevK)
        self.nextYs.append(nextY)

        for i in range(nextY.size(0)):
            if nextY[i] == tokenizer.eos_token_id:
                self.finished.append((self.scores[i], len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if nextY[0] == tokenizer.eos_token_id:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.beam_size

    def getFinal(self) -> List[Tuple[Tensor, int, int]]:
        if not self.finished:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        else:
            self.finished.sort(key=lambda a: -a[0])

        if len(self.finished) < self.beam_size:
            nextY = self.nextYs[-1]
            unfinished = [(self.scores[i], len(self.nextYs) - 1, i)
                          for i in range(nextY.size(0))
                          if nextY[i] != tokenizer.eos_token_id]
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[: self.beam_size - len(self.finished)]

        return self.finished[: self.beam_size]

    def getHyp(self, beam_res: List[Tuple[Tensor, int, int]]):
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
        def f(pred):
            tokens = []
            for tok in pred:
                if tok == tokenizer.eos_token_id:
                    break
                tokens.append(tok)
            return tokens

        return [f(pred) for pred in preds]

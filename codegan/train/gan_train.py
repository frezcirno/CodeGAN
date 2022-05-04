import argparse
import logging
import os
import sys
from typing import Literal
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from codegan.train.dis_train import DisTrainer

from codegan.train.gen_train import GenTrainer

import tokenizer
from ..generator import Generator
from ..discriminator import Discriminator
from ..utils import set_seed
from ..utils.dist import is_distributed, local_rank
from ..utils.meter import MaxMeter, BatchAvgMeter, MinMeter
from .utils import Trainer, add_general_arguments, eval_dis_acc, evaluate_metrics, eval_gen_loss, fakegen2, init_run_dir, is_notebook, save_model, setup_gpu, setup_logging

logger = logging.getLogger(__name__)


class GanTrainer(Trainer):
    def __init__(self, args, run_dir, load_path, dis_load_path, device, dis_device, parallel=True):
        super().__init__(args, run_dir, device)

        self.args = args
        self.model = Generator(args.hidden_size, args.vocab_size, args.beam_size, args.tgt_max_len)

        if load_path:
            self.model = self.load_model(self.model, load_path, device)

        self.model.to(device)

        if parallel:
            self.model = self.build_parallel(
                self.model,
                find_unused_parameters=False,
                device=device
            )

        self.dis = Discriminator(args.src_max_len, args.tgt_max_len, args.vocab_size, args.dis_hidden_size)

        for p in self.dis.embedding.parameters():
            p.requires_grad = False

        if dis_load_path:
            self.dis = self.load_model(self.dis, dis_load_path, dis_device)

        self.dis.to(dis_device)

        if parallel:
            self.dis = self.build_parallel(
                self.dis,
                find_unused_parameters=False,
                device=dis_device
            )

        self.dis_device = dis_device

        self.prepare_checkpoint()

        self.best_bleu = MaxMeter()
        self.best_acc = MaxMeter()

    @classmethod
    def modify_weights(cls, weights: Tensor) -> Tensor:
        weights = GenTrainer.modify_weights(weights)
        weights = DisTrainer.modify_weights(weights)
        return weights

    def to_dis_device(self, x: Tensor):
        return x.to(self.dis_device)

    def gen_train(self):
        self.model.train()

    def gen_eval(self):
        self.model.eval()

    def dis_train(self):
        self.dis.train()

    def dis_eval(self):
        self.dis.eval()

    def get_reward(
        self,
        source_ids,
        source_mask,
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
            give pre_target_ids[:tgt_max_len-1], predict, get reward
        Input:
            pre_target_ids: [batch_size x target_length], with bos!!
        Outputs:
            rewards: [batch_size x tgt_max_len]
                rewards[i][0] is empty
                rewards[i][j]: the reward of using word Seq[j] as the next of Seq[0..j-1].
        """
        self.dis_eval()
        batch_size = source_ids.size(0)

        rewards = torch.zeros(self.args.tgt_max_len, batch_size,
                              dtype=torch.float, device=self.dis_device)
        with torch.no_grad():
            model = getattr(self.model, 'module', self.model)
            memory, memory_key_padding_mask = model.encode(source_ids, source_mask)
            for init_given_num in range(2, self.args.tgt_max_len):  # ignore bos_token
                if not any(pre_target_mask[:, init_given_num]):
                    break
                init_target_ids = pre_target_ids[:, :init_given_num]
                for _ in range(rollnum):  # rollout times, mean() later
                    target_ids, _ = self.model(
                        memory=memory,
                        memory_key_padding_mask=memory_key_padding_mask,
                        target_ids=init_target_ids,
                        init_given_num=init_given_num,
                    )
                    # pred: [0-1] prods
                    pred = self.dis(self.to_dis_device(source_ids),
                                    self.to_dis_device(target_ids))
                    # pred = pred.cpu()
                    rewards[init_given_num - 1] += pred

            # [batch_size]
            pred = self.dis(self.to_dis_device(source_ids),
                            self.to_dis_device(pre_target_ids))
            rewards[self.args.tgt_max_len - 1] += pred * rollnum

        # rewards: [batch_size x tgt_max_len]
        rewards = rewards.transpose(1, 0).contiguous()
        rewards = rewards * pre_target_mask
        rewards = rewards / (1.0 * rollnum)
        return rewards.to(pre_target_mask.device)

    def prepare_checkpoint(self):
        self.register_path("latest", "gan_gen.bin")
        self.register_path("latest_dis", "gan_dis.bin")
        self.register_path("best_loss", "gan_gen_bestloss_%f.bin")
        self.register_path("best_loss_dis", "gan_dis_bestloss_%f.bin")
        self.register_path("best_bleu", "gan_gen_bestbleu_%f.bin")
        self.register_path("best_acc_dis", "gan_dis_bestacc_%f.bin")
        self.register_path('output_file', f"gan_{local_rank()}.output" if is_distributed() else "gan.output")
        self.register_path('gold_file', f"gan_{local_rank()}.gold" if is_distributed() else "gan.gold")

    def save_checkpoint_gen(self, type: Literal['latest|best_loss|best_bleu'], *args):
        path = self.get_path(type, *args)
        save_model(self.model, path)

    def save_checkpoint_dis(self, type: Literal['latest|best_loss|best_acc'], *args):
        dis_path = self.get_path(type + "_dis", *args)
        save_model(self.dis, dis_path)

    def __prepare_optimizer_gen(self):
        # TODO: Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.opt = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )

        t_total = self.args.g_steps * self.train_epochs
        if self.args.teach:
            t_total = t_total * 2

        logger.info("+ Total train steps = %d", t_total)
        logger.info("+ Warmup steps = %d", int(t_total * 0.1))

        self.sch = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=int(t_total * 0.1),
                                                   num_training_steps=t_total)

    def __prepare_optimizer_dis(self):
        self.dis_opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.dis.parameters()),
            lr=self.args.dis_learning_rate,
            eps=self.args.dis_adam_epsilon
        )

    def train_step_gen(self, batch):
        source_ids = self.to_model_device(batch[0])
        source_mask = self.to_model_device(batch[1])

        pre_target_ids, pre_target_mask = self.model(source_ids, source_mask)
        rewards = self.get_reward(
            source_ids, source_mask,
            pre_target_ids, pre_target_mask,
            rollnum=self.args.rollnum
        )
        # get pg_loss
        loss: Tensor = self.model(
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=pre_target_ids,
            rewards=rewards
        )

        if loss.size():
            loss = loss.mean()

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.sch.step()

        return loss.item(), source_ids.size(0)

    def train_step_gen_tf(self, batch):
        source_ids = self.to_model_device(batch[0])
        source_mask = self.to_model_device(batch[1])
        target_ids = self.to_model_device(batch[2])
        target_mask = self.to_model_device(batch[3])

        # [batch_size x source_length x args.hidden_size]

        rewards = torch.ones_like(target_ids) * target_mask
        tloss = self.model(source_ids, source_mask, target_ids, rewards=rewards)

        if tloss.size():
            tloss = tloss.mean()

        tloss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.sch.step()

        return tloss.item(), source_ids.size(0)

    def train_step_dis(self, dataloader, sampler):
        # Train dis for k epochs
        for d_epoch in trange(self.args.d_epochs, desc="d-step epoch"):
            if is_distributed():
                sampler.set_epoch(d_epoch)

            loss_meter = BatchAvgMeter()
            with tqdm(dataloader, "BCEloss 0.0000", dynamic_ncols=True) as bar:
                for batch in bar:
                    source_ids = self.to_dis_device(batch[0])
                    target_ids = self.to_dis_device(batch[1])
                    labels = self.to_dis_device(batch[2])

                    prob = self.dis(source_ids, target_ids)
                    loss = F.binary_cross_entropy(prob, labels, reduction='sum')
                    # mean() to average on multi-gpu
                    if loss.size():
                        loss = loss.mean()

                    loss.backward()
                    self.dis_opt.step()
                    self.dis_opt.zero_grad()

                    d_loss_avg = loss_meter.update(loss.item(), labels.size(0))
                    bar.set_description(f"BCEloss {d_loss_avg:.4f}")

            logger.info("d-epoch %d train BCEloss %f", d_epoch, d_loss_avg)

    def train_epoch_gen(self, train_iter):
        # G-steps
        self.gen_train()
        self.dis_eval()
        g_loss = BatchAvgMeter()
        g_tloss = BatchAvgMeter()
        with trange(self.args.g_steps, desc="g-step 00.0000 00.0000") as g_step_bar:
            g_step = iter(g_step_bar)
            while True:
                try:
                    next(g_step)
                except StopIteration:
                    break

                batch = next(train_iter)
                batch_loss, batch_samples = self.train_step_gen(batch)
                g_loss_avg = g_loss.update(batch_loss, batch_samples)

                if self.args.teach:
                    tloss, tsamples = self.train_step_gen_tf(batch)
                    g_tloss_avg = g_tloss.update(tloss, tsamples)

                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} {g_tloss_avg:.4f}")
                else:
                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} -")

        logger.info("g-step train avg loss (gan only): %f", g_loss_avg)
        if self.args.teach:
            logger.info("g-step train avg loss (teach only): %f", g_tloss_avg)
            logger.info("g-step train avg loss: %f", (g_loss_avg + g_tloss_avg) / 2)
            # writer.add_scalar("GAN/Gen/Loss", (g_loss_avg + g_tloss_avg) / 2, self.epoch)
        else:
            # writer.add_scalar("GAN/Gen/Loss", g_loss_avg, self.epoch)
            pass

    def train_epoch_dis(self, train_dataset):
        self.gen_eval()
        self.dis_train()
        for d_step in trange(self.args.d_steps, desc="d-step"):
            # (re)generate fake dataset
            dis_train_dataset = fakegen2(self.model, self.device, train_dataset,
                                         self.args.gen_num_train_batchs, self.args.gen_batch_size,
                                         self.args.gen_beam_search, self.num_workers)

            dataloader, sampler = self.train_dataloader(dis_train_dataset)

            self.train_step_dis(dataloader, sampler)

    def train(self, train_dataset, valid_dataset, test_dataset):
        self.__prepare_optimizer_gen()
        self.__prepare_optimizer_dis()
        # '''
        # In theory, the whole dataset should be used to train the discriminator,
        # but that's very slow. So a subset is used. The subset won't change
        # with d-step.
        # '''
        # train_subset = SamplingSubset(train_dataset, self.args.gen_num_train_batchs * self.args.gen_batch_size)

        '''
        Change data for pg train every epoch.
        '''
        dataloader, sampler = self.train_dataloader(train_dataset)
        train_iter = iter(dataloader)

        for self.epoch in trange(self.train_epochs, desc="Epoch"):
            # Not need, because one epoch here doesn't exhaust all samples
            # if is_distributed():
            #     gan_train_sampler.set_epoch(self.epoch)
            self.train_epoch_gen(train_iter)
            self.save_checkpoint_gen('latest')

            self.eval_epoch_gen(valid_dataset, test_dataset)

            self.train_epoch_dis(train_dataset)
            self.save_checkpoint_dis('latest')

            self.eval_epoch_dis(valid_dataset, test_dataset)

    def eval(self, valid_dataset, test_dataset):
        # loss = self.eval_loss(valid_dataset)
        # logger.info("+ loss = %f", loss)
        metrics = self.eval_metrics(test_dataset)
        logger.info("+ metrics = %s", metrics)
        acc = self.eval_dis_acc(test_dataset)
        logger.info("+ Accurary = %f", acc)

    def eval_dis_acc(self, test_dataset):
        test_dataset = fakegen2(self.model, self.device, test_dataset,
                                self.args.gen_num_test_batchs, self.args.gen_batch_size,
                                self.args.gen_beam_search, self.num_workers)

        return eval_dis_acc(
            self.dis,
            self.dis_device,
            test_dataset,
            self.eval_batch_size,
            self.num_workers,
        )

    def eval_epoch_gen(self, valid_dataset, test_dataset):
        # loss = self.eval_loss(valid_dataset)
        # logger.info(f"+ Eval loss: {loss:.5f}")
        # best_loss = self.best_loss.update(loss)
        # if self.best_loss.is_best():
        #     self.save_checkpoint_gen('best_loss', best_loss)

        metrics = self.eval_metrics(test_dataset)
        logger.info("+ metrics = %s", metrics)
        dev_bleu = metrics['bleu4']
        best_bleu = self.best_bleu.update(dev_bleu)
        if self.best_bleu.is_best():
            self.save_checkpoint_gen('best_bleu', best_bleu)

    def eval_loss(self, valid_dataset):
        # Eval G with dev dataset
        self.gen_eval()

        return eval_gen_loss(
            self.model,
            self.device,
            valid_dataset,
            self.batch_size,
            self.num_workers
        )

    def eval_epoch_dis(self, valid_dataset, test_dataset):
        acc = self.eval_dis_acc(test_dataset)
        logger.info(f"+ D accuracy: {acc:.5f}")
        best_acc = self.best_acc.update(acc)
        if self.best_acc.is_best():
            self.save_checkpoint_dis('best_acc', best_acc)

    def eval_metrics(self, test_dataset):
        self.gen_eval()

        return evaluate_metrics(
            self.model,
            self.device,
            test_dataset,
            self.get_path("output_file"),
            self.get_path("gold_file"),
            self.eval_batch_size,
            self.num_workers
        )

    @classmethod
    def add_arguments(cls, parser):
        GenTrainer.add_arguments(parser)

        # --load_path is reserved for the main model (the G)
        parser.add_argument('--dis_load_path', type=str)
        parser.add_argument("--dis_hidden_size", type=int, default=768)
        parser.add_argument("--dis_learning_rate", type=float, default=5e-5)
        parser.add_argument("--dis_adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--dis_weight_decay", type=float, default=0.0)

        parser.add_argument("--rollnum", type=int, default=20)
        parser.add_argument("--g_steps", type=int, default=100, help="Generator train steps, one batch one step.")
        parser.add_argument("--teach", action="store_true", help="Use teacher forcing after every step.")
        parser.add_argument("--d_steps", type=int, default=5,
                            help="Discriminator train steps, do gan_d_sample x d_epochs samples one step.")
        parser.add_argument('-k', "--d_epochs", type=int, default=6)
        parser.add_argument('--d_batch_size', type=int, default=32)

        parser.add_argument('--gen_batch_size', type=int, default=256)
        parser.add_argument('--gen_num_train_batchs', type=int, default=2000)
        parser.add_argument('--gen_num_valid_batchs', type=int, default=200)
        parser.add_argument('--gen_num_test_batchs', type=int, default=200)
        parser.add_argument('--gen_beam_search', action='store_true')


if __name__ == '__main__':
    run_dir = init_run_dir("gan")
    setup_logging(run_dir)

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    Trainer.add_arguments(parser)
    GanTrainer.add_arguments(parser)

    TRAIN_ARGS = '''--help'''.split()
    args = parser.parse_args(TRAIN_ARGS if is_notebook() else sys.argv[1:])

    logger.info(" ".join(sys.argv))
    logger.info(str(args))

    set_seed(args.seed)

    if not torch.cuda.is_available():
        logger.error("cuda is not available")
        exit(-1)

    _device, _ = setup_gpu(args.device_ids, args.occupy)
    logger.info(f"Using device {_device}")

    train_dataset, valid_dataset, test_dataset = torch.load(args.data)
    logger.info("train dataset: %d samples", len(train_dataset))
    logger.info("valid dataset: %d samples", len(valid_dataset))
    logger.info("test dataset: %d samples", len(test_dataset))

    trainer = GanTrainer(args, run_dir, args.load_path, args.dis_load_path, _device, _device)
    trainer.run(train_dataset, valid_dataset, test_dataset)

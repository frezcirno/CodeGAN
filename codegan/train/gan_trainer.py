import logging
from typing import Literal
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .utils import Trainer, eval_gen_bleu

from ..bleu import bleu
from ..tokenize import tensors_to_text
from ..utils.dist import is_distributed
from ..utils.meter import MaxMeter, axMeter, BatchAvgMeter, MinMeter

logger = logging.getLogger(__name__)


class Rollout:
    def __init__(self, gen, dis, max_length):
        self.gen = gen
        self.dis = dis
        self.max_length = max_length
        self.gen_device = gen.device
        self.dis_device = dis.device

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
                target_ids, target_mask = self.gen(
                    context,
                    memory_key_padding_mask,
                    target_ids=pre_target_ids,
                    rollout=True,
                    init_given_num=init_given_num,
                )
                with torch.no_grad():
                    # pred: [0-1] prods
                    pred = self.dis(
                        source_ids.to(self.dis_device),
                        target_ids.to(self.dis_device)
                    )
                # pred = pred.cpu()
                rewards[init_given_num - 1] += pred

            with torch.no_grad():
                pred = self.dis(
                    source_ids.to(self.dis_device),
                    pre_target_ids.to(self.dis_device)
                )
            # pred = pred.cpu()
            # [batch_size]
            rewards[self.max_length - 1] += pred

        # rewards: [batch_size x max_length]
        rewards = rewards.permute([1, 0]).contiguous()
        rewards = rewards * pre_target_mask
        rewards = rewards / (1.0 * rollnum)
        return rewards.to(pre_target_mask.device)


class GanTrainer(Trainer):
    def __init__(self, args, _gen, gen, _dis, dis):
        self.args = args
        self._gen = _gen
        self.gen = gen
        self._dis = _dis
        self.dis = dis
        self.rollout = Rollout(gen, dis, args.tgt_max_len)

        self.gen_device = _gen.device
        self.dis_device = _dis.device

        for p in self._dis.src_embedding.parameters():
            p.requires_grad = False
        for p in self._dis.tgt_embedding.parameters():
            p.requires_grad = False

        self.__prepare_path()
        self.__prepare_eval()
        self.__prepare_optimizer_gen()
        self.__prepare_optimizer_dis()

        self.epoch = -1

    def to_gen_device(self, x):
        return x.to(self.gen_device)

    def to_dis_device(self, x):
        return x.to(self.dis_device)

    def gen_to_cpu(self):
        self.gen.to("cpu")

    def gen_to_gpu(self):
        self.gen.to(self.gen_device)

    def dis_to_cpu(self):
        self.dis.to("cpu")

    def dis_to_gpu(self):
        self.dis.to(self.dis_device)

    def gen_train(self):
        self.gen.train()

    def gen_eval(self):
        self.gen.eval()

    def dis_train(self):
        self.dis.train()

    def dis_eval(self):
        self.dis.eval()

    def save_models(self, type: Literal['latest|best_loss|best_bleu'], val=0):
        if type == "latest":
            gen_path = self.checkpoints['latest_gen']
            dis_path = self.checkpoints['latest_dis']
        elif type == "best_loss":
            gen_path = self.checkpoints['best_loss_gen'] % (self.epoch, val)
            dis_path = self.checkpoints['best_loss_dis'] % (self.epoch, val)
        else:
            gen_path = self.checkpoints['best_bleu_gen'] % (self.epoch, val)
            dis_path = self.checkpoints['best_bleu_dis'] % (self.epoch, val)

        save_model(self._gen, gen_path)
        save_model(self._dis, dis_path)

    def __prepare_path(self):
        from os.path import join as path_join

        self.checkpoints = {
            "latest_gen": path_join(self.args.ckpt_dir, "gan_gen.bin"),
            "latest_dis": path_join(self.args.ckpt_dir, "gan_dis.bin"),
            "best_loss_gen": path_join(self.args.ckpt_dir, "gan_gen_bestloss_%d_%f.bin"),
            "best_loss_dis": path_join(self.args.ckpt_dir, "gan_dis_bestloss_%d_%f.bin"),
            "best_bleu_gen": path_join(self.args.ckpt_dir, "gan_gen_bestbleu_%d_%f.bin"),
            "best_bleu_dis": path_join(self.args.ckpt_dir, "gan_dis_bestbleu_%d_%f.bin"),
            "bleu_output": path_join(self.args.ckpt_dir, "gan_dev_%d.output"),
            "bleu_gold": path_join(self.args.ckpt_dir, "gan_dev_%d.gold")
        }

        logger.info("Checkpoint paths: %s", self.checkpoints)

    def __prepare_optimizer_gen(self):
        logger.info("+ Generator learning rate = %s", self.args.gen_learning_rate)
        logger.info("+ Generator adam epsilon = %e", self.args.gen_adam_epsilon)
        # TODO: Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.gen.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.gen.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.g_opt = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.gen_learning_rate,
            eps=self.args.gen_adam_epsilon,
        )

        t_total = self.args.gan_g_steps * self.args.gan_train_epochs
        if self.args.gan_teach:
            t_total = t_total * 2

        logger.info("+ Total train steps = %d", t_total)
        logger.info("+ Warmup steps = %d", int(t_total * 0.1))

        self.g_sch = get_linear_schedule_with_warmup(
            self.g_opt, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total)

    def __prepare_optimizer_dis(self):
        logger.info("+ Discriminator learning rate = %s", self.args.dis_learning_rate)
        logger.info("+ Discriminator adam epsilon = %e", self.args.dis_adam_epsilon)
        self.d_opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.dis.parameters()),
            lr=self.args.dis_learning_rate,
            eps=self.args.dis_adam_epsilon
        )

    def train_step_gen(self, batch):
        source_ids = self.to_gen_device(batch[0])
        source_mask = self.to_gen_device(batch[1])

        """672 MiB"""
        context, memory_key_padding_mask = self._gen.encode(source_ids, source_mask)
        # [batch_size x source_length x args.hidden_size]

        pre_target_ids, pre_target_mask = self.gen(context, memory_key_padding_mask)
        rewards = self.rollout.get_reward(
            source_ids,
            context,
            memory_key_padding_mask,
            pre_target_ids,
            pre_target_mask,
            rollnum=self.args.gan_rollnum,
        )
        # get pg_loss
        loss = self.gen(context, memory_key_padding_mask, pre_target_ids, rewards=rewards)

        if loss.size():
            loss = loss.mean()

        loss.backward()
        self.g_opt.step()
        self.g_opt.zero_grad()
        self.g_sch.step()

        return loss.item(), source_ids.size(0)

    def train_step_gen_tf(self, batch):
        source_ids = self.to_gen_device(batch[0])
        source_mask = self.to_gen_device(batch[1])
        target_ids = self.to_gen_device(batch[2])
        target_mask = self.to_gen_device(batch[3])

        context, memory_key_padding_mask = self._gen.get_context(source_ids, source_mask)
        # [batch_size x source_length x args.hidden_size]

        rewards = torch.ones_like(target_ids) * target_mask
        tloss = self.gen(context, memory_key_padding_mask, target_ids, rewards=rewards)

        if tloss.size():
            tloss = tloss.mean()

        tloss.backward()
        self.g_opt.step()
        self.g_opt.zero_grad()
        self.g_sch.step()

        return tloss.item(), source_ids.size(0)

    def train_step_dis(self, dis_train_dataset):
        dataloader = DataLoader(
            dis_train_dataset,
            batch_size=self.args.gan_d_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        # Train dis for k epochs
        for d_epoch in trange(self.args.gan_d_epochs, desc="d-step epoch"):
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
                    self.d_opt.step()
                    self.d_opt.zero_grad()

                    d_loss_avg = loss_meter.update(loss.item(), labels.size(0))
                    bar.set_description(f"BCEloss {d_loss_avg:.4f}")

            logger.info("d-epoch train BCEloss %f", d_loss_avg)

    def train_epoch_gen(self, train_iter):
        # G-steps
        self.gen_to_gpu()
        self.gen_train()
        self.dis_to_gpu()
        self.dis_eval()
        g_loss = BatchAvgMeter()
        g_tloss = BatchAvgMeter()
        with trange(self.args.gan_g_steps, desc="g-step 00.0000 00.0000") as g_step_bar:
            g_step = iter(g_step_bar)
            while True:
                try:
                    next(g_step)
                except StopIteration:
                    break

                batch = next(train_iter)
                batch_loss, batch_samples = self.train_step_gen(batch)
                g_loss_avg = g_loss.update(batch_loss, batch_samples)

                if self.args.gan_teach:
                    tloss, tsamples = self.train_step_gen_tf(batch)
                    g_tloss_avg = g_tloss.update(tloss, tsamples)

                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} {g_tloss_avg:.4f}")
                else:
                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} -")

        logger.info("g-step train avg loss (gan only): %f", g_loss_avg)
        if self.args.gan_teach:
            logger.info("g-step train avg loss (teach only): %f", g_tloss_avg)
            logger.info("g-step train avg loss: %f", (g_loss_avg + g_tloss_avg) / 2)
            # writer.add_scalar("GAN/Gen/Loss", (g_loss_avg + g_tloss_avg) / 2, self.epoch)
            pass
        else:
            # writer.add_scalar("GAN/Gen/Loss", g_loss_avg, self.epoch)
            pass

    def sample_for_dis(self, train_subset):
        '''
        Notice: The order of the return dataset is equal to the train_subset,
        as shuffle is used in train_step_dis(), so not shuffle here.
        '''
        if is_distributed():
            sampler = DistributedSampler(train_subset)
        else:
            sampler = SequentialSampler(train_subset)

        dataloader = DataLoader(
            train_subset,
            batch_size=self.args.fakegen_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

        logger.info("+ Gen fake: batch_size = %d, samples ~= %d", self.args.fakegen_batch_size,
                    self.args.fakegen_batch_size * len(dataloader))
        return make_gan_dataset(self._gen, dataloader)

    def train_epoch_dis(self, train_subset):
        # D-steps
        self.gen_eval()
        self.dis_train()
        for d_step in trange(self.args.gan_d_steps, desc="d-step"):
            # (re)generate fake dataset
            self.gen_to_gpu()
            dis_train_dataset = self.sample_for_dis(train_subset)
            self.gen_to_cpu()

            self.dis_to_gpu()
            self.train_step_dis(dis_train_dataset)
            self.dis_to_cpu()

    def train(self, train_dataset, valid_dataset, bleu_dataset):
        logger.info("Do GAN train:")

        logger.info("+ train dataset = %d", len(train_dataset))
        logger.info("+ valid dataset = %d", len(valid_dataset))
        logger.info("+ bleu dataset = %d", len(bleu_dataset))

        logger.info("+ g-steps = %d", self.args.gan_g_steps)
        logger.info("+ Teacher forcing = %s", self.args.gan_teach)
        logger.info("+ d-steps = %d", self.args.gan_d_steps)
        logger.info("+ d-sample = %d", self.args.gan_d_sample)
        logger.info("+ d-epochs = %d", self.args.gan_d_epochs)
        logger.info("+ Rollout num = %d", self.args.gan_rollnum)
        logger.info("+ GAN batch size = %d", self.args.gan_batch_size)

        '''
        In theory, the whole dataset should be used to train the discriminator,
        but that's very slow. So a subset is used. The subset won't change
        with d-step.
        '''
        train_subset = SamplingSubset(train_dataset, self.args.gan_d_sample)

        '''
        Change data for pg train every epoch.
        '''
        if is_distributed():
            gan_train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            gan_train_sampler = RandomSampler(train_dataset)

        gan_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.gan_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=gan_train_sampler,
        )
        train_iter = iter(gan_dataloader)

        for self.epoch in trange(self.args.gan_train_epochs, desc="Epoch"):
            # Not need, because one epoch here doesn't exhaust all samples
            # if is_distributed():
            #     gan_train_sampler.set_epoch(self.epoch)
            self.train_epoch_gen(train_iter)
            self.train_epoch_dis(train_subset)
            self.save_models('latest')
            self.eval_epoch(valid_dataset, bleu_dataset)

    def __prepare_eval(self):
        self.best_loss = MinMeter()
        self.best_bleu = MaxMeter()

    def eval(self, valid_dataset, bleu_dataset):
        self.dis_to_cpu()
        self.gen_eval()
        self.gen_to_gpu()
        self.eval_gen_loss(valid_dataset)
        self.eval_gen_bleu(bleu_dataset)
        self.eval_dis_acc(valid_dataset)

    def eval_dis_acc(self, valid_dataset):
        dis_valid_dataset = fakegen(
            self.args,
            valid_dataset,
            self.args.dis_valid_sample,
            self._gen,
            train=False,
        )

        acc_meter = BatchAvgMeter()
        dataloader = DataLoader(
            dis_valid_dataset,
            batch_size=self.args.dis_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.dis_eval()
        self.dis_to_gpu()
        with torch.no_grad():
            with tqdm(dataloader, "Accu 00.0000", dynamic_ncols=True) as bar:
                for batch in bar:
                    source_ids = self.to_dis_device(batch[0])
                    target_ids = self.to_dis_device(batch[1])
                    labels = batch[2]
                    prob = self.dis(source_ids, target_ids).to("cpu").gt(0.5)
                    right = labels.eq(prob).sum()
                    total = source_ids.size(0)
                    acc = acc_meter.update(right, total)
                    bar.set_description(f"Accu {acc:.4f}")

        logger.info("+ eval acc = %f", acc)
        return acc

    def eval_epoch(self, valid_dataset, bleu_dataset):
        self.dis_to_cpu()
        self.gen_eval()
        self.gen_to_gpu()

        loss = self.eval_gen_loss(valid_dataset)
        if self.best_loss.update(loss) == loss:
            logger.info("+ Best loss !!")
            self.save_models('best_loss', self.best_loss.get())

        dev_bleu = self.eval_gen_bleu(bleu_dataset)
        if self.best_bleu.update(dev_bleu) == dev_bleu:
            logger.info("+ Best bleu !!")
            self.save_models('best_bleu', self.best_bleu.get())

        self.eval_dis_acc(valid_dataset)

    def eval_gen_loss(self, valid_dataset):
        # Eval G with dev dataset
        logger.info("+ Valid dataset = %d", len(valid_dataset))
        logger.info("+ batch size = %d", self.args.eval_batch_size)
        logger.info("+ num workers = %d", self.args.num_workers)

        loss = self.get_loss(valid_dataset)
        logger.info("+ eval loss = %f", loss)
        return loss

    def eval_gen_bleu(self, bleu_dataset):
        logger.info("+ Bleu dataset = %d", len(bleu_dataset))
        logger.info("+ batch size = %d", self.args.eval_batch_size)
        logger.info("+ num workers = %d", self.args.num_workers)

        dev_bleu = self.get_bleu(
            bleu_dataset,
            self.checkpoints["bleu_output"] % (self.epoch, ),
            self.checkpoints["bleu_gold"] % (self.epoch, ),
        )
        logger.info("+ bleu-4 = %f", dev_bleu)
        return dev_bleu

    def get_loss(self, valid_dataset):
        loss_meter = BatchAvgMeter()
        gan_valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        for batch in tqdm(gan_valid_dataloader, "validation", dynamic_ncols=True):
            source_ids = batch[0].to(self.gen_device)
            source_mask = batch[1].to(self.gen_device)
            target_ids = batch[2].to(self.gen_device)
            target_mask = batch[3].to(self.gen_device)

            with torch.no_grad():
                context, memory_key_padding_mask = self._gen.get_context(
                    source_ids, source_mask
                )
                # [batch_size x source_length x args.hidden_size]

                loss, num, _ = self.gen(
                    context, memory_key_padding_mask, target_ids, target_mask
                )
                loss *= num
                if loss.size():  # is multi-gpu
                    loss = loss.sum()
                    num = num.sum()
            eval_loss = loss_meter.update(loss.item(), num.item())

        return eval_loss

    def get_bleu(self, test_dataset, gan_output_file, gan_gold_file):
        # Save best checkpoint for best bleu
        logger.info("Calculate bleu-4:")
        logger.info("+ gan_output_file = %s", gan_output_file)
        logger.info("+ gan_gold_file = %s", gan_gold_file)

        return eval_gen_bleu(
            self.gen,
            test_dataset,
            self.checkpoints["bleu_output"],
            self.checkpoints["bleu_gold"],
            self.args.eval_batch_size,
        )

    @classmethod
    def add_argument(cls, parser):
        # Gan
        parser.add_argument("--do_gan_train", action="store_true")
        parser.add_argument("--do_gan_eval", action="store_true")
        parser.add_argument("--gan_batch_size", type=int, default=16)
        parser.add_argument("--gan_train_epochs", type=int, default=30)
        parser.add_argument("--gan_rollnum", type=int, default=20)
        parser.add_argument("--gan_g_steps", type=int, default=100, help="Generator train steps, one batch one step.")
        parser.add_argument("--gan_teach", action="store_true", help="Use teacher forcing after every step.")
        parser.add_argument(
            "--gan_d_steps", type=int, default=5,
            help="Discriminator train steps, do gan_d_sample x gan_d_epochs samples one step."
        )
        parser.add_argument("--gan_d_sample", type=int, default=1000)
        parser.add_argument("--gan_d_epochs", type=int, default=6)
        parser.add_argument("--gan_d_batch_size", type=int, default=64)

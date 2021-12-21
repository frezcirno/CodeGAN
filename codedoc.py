# %%
import time
import math
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from codedoc_model import *


# %%
args = {
    "config_name": "",
    "tokenizer_name": "",
    "model_name": "",
}

# %%
java_data = pd.read_pickle("jd.processed.pkl")

java_data_train = java_data[java_data["partition"] == "train"]
java_data_test = java_data[java_data["partition"] == "test"]
java_data_valid = java_data[java_data["partition"] == "valid"]

# %%
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)

config = RobertaConfig.from_pretrained(args.config_name)
model = RobertaModel.from_pretrained(args.tokenizer_name)
tokenizer = RobertaTokenizer.from_pretrained()

# %%
# OOV问题：
# fastPathOrderedEmit -> fast path ordered emit
# javalang -> identifier -> 驼峰分词
# **BPE** -> 词根分词
# 减少vocab的数量

# 代码，注释转小写

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8


def yield_tokens(column):
    for tokens in column:
        yield tokens


code_vocab = build_vocab_from_iterator(
    yield_tokens(java_data["code_tokens"]),
    min_freq=10,
    specials=["<pad>", "<bos>", "<eos>"],
)
doc_vocab = build_vocab_from_iterator(
    yield_tokens(java_data["docstring_tokens"]), specials=["<pad>", "<bos>", "<eos>"],
)

PAD_IDX = code_vocab["<pad>"]
BOS_IDX = code_vocab["<bos>"]
EOS_IDX = code_vocab["<eos>"]

print(len(code_vocab))  # 5w左右正常
print(len(doc_vocab))


def make_generator(df):
    for _, s in df.iterrows():
        yield s["code_tokens"], s["docstring_tokens"]


def make_tensor(df_gen):
    data = []
    for code_tokens, docstring_tokens in df_gen:
        code_tensor = torch.tensor(
            [code_vocab[token] for token in code_tokens if token in code_vocab],
            dtype=torch.long,
        )
        doc_tensor = torch.tensor(
            [doc_vocab[token] for token in docstring_tokens], dtype=torch.long
        )
        data.append((code_tensor, doc_tensor))
    return data


def generate_batch(data_batch):
    code_batch, doc_batch = [], []
    for code, doc in data_batch:
        code_batch.append(
            torch.cat([torch.tensor([BOS_IDX]), code, torch.tensor([EOS_IDX])], dim=0)
        )
        doc_batch.append(
            torch.cat([torch.tensor([BOS_IDX]), doc, torch.tensor([EOS_IDX])], dim=0)
        )
    code_batch = pad_sequence(code_batch, padding_value=PAD_IDX)  # truncate -> 300
    doc_batch = pad_sequence(doc_batch, padding_value=PAD_IDX)
    return code_batch, doc_batch


# %%
train_iter = DataLoader(
    make_tensor(make_generator(java_data_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=generate_batch,
)
valid_iter = DataLoader(
    make_tensor(make_generator(java_data_valid)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=generate_batch,
)
test_iter = DataLoader(
    make_tensor(make_generator(java_data_test)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=generate_batch,
)

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
model = nn.DataParallel(model)
model = model.to(device)

# print(model.apply(init_weights))
model.load_state_dict(torch.load("codedoc.pt"))

optimizer = optim.Adam(model.parameters())

print(f"The model has {count_parameters(model):,} trainable parameters")


criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
):
    model.train()

    epoch_loss = 0

    for code_bat, doc_bat in tqdm.tqdm(iterator):
        code_bat = code_bat.to(device)
        doc_bat = doc_bat.to(device)

        optimizer.zero_grad()
        output = model(code_bat, doc_bat)

        output = output[1:].view(-1, output.shape[-1])
        doc_bat = doc_bat[1:].view(-1)

        loss = criterion(output, doc_bat)
        loss.requires_grad_(True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 80
CLIP = 1

best_valid_loss = float("inf")

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    epoch_mins, epoch_secs = epoch_time(start_time, time.time())

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")
    torch.save(model.state_dict(), "codedoc.pt")

test_loss = evaluate(model, test_iter, criterion)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")

# %%

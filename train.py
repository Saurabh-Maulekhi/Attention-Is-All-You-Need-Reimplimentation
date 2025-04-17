from model import build_transformer
from dataset import BilingualDataset, casual_mask
from config import get_config, get_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
import os
from pathlib import Path

# Huggingface datasets and tokenizers
# from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# data Analyssis and manipulation
import pandas as pd


dataset = pd.read_csv("dataset/newdata.csv")
dataset.head()

# Dropping NA rows
dataset = dataset.dropna()

# Renaming
dataset.rename(columns={'Unnamed: 0': 'id'}, inplace=True)


def get_ds(config):
    """
    config: Configuration of model
    return: train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    """
    # Loading the train portion of the  dataset.
    # The Language pairs will be defined in the 'config' dictionary we will build later

    ds_raw = dataset # dataset , csv file

    # Building or loading tokenizer for both the source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Splitting the dataset for training and validation
    train_ds_size = int(0.9 * len(ds_raw)) # 90 % for training
    val_ds_size = len(ds_raw) - train_ds_size # 10% for validation

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # Randomly splitting the dataset

    # Processing data with the BilingualDataset class
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and target languages
    max_len_src = 0
    max_len_tgt = 0

    for index, row in ds_raw.iterrows():
        src_sentence = row[config['lang_src']]
        tgt_sentence = row[config['lang_tgt']]

        src_ids = tokenizer_src.encode(src_sentence).ids
        tgt_ids = tokenizer_tgt.encode(tgt_sentence).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # Creating dataloaders for the training and validadion sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the eos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        # Select the token with the max probablity (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                                  dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def get_all_attention_maps(attn_type: str, layers: list[int], heads: list[int], row_tokens:list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)

def get_or_build_tokenizer(config, ds, lang):
    """
    config: the dictionary that have configuration details of our model
    ds : dataset
    lang : language of which we are getting or building Tokenizer
    return : Tokenizer file of given lang (language)
    """

    tokenizer_path = Path(config['tokenizer_file'].format(lang))  # Path of our Tokenizer's file path

    tokenizer = Tokenizer.from_file(str(tokenizer_path))  # getting tokenizer from the file

    return tokenizer

dataset = pd.read_csv("dataset/newdata.csv")
dataset.head()

# Dropping NA rows
dataset = dataset.dropna()

# Renaming
dataset.rename(columns={'Unnamed: 0': 'id'}, inplace=True)


# We pass as parameters the config dictionary, the length of the vocabylary of the source language and the target language
def get_model(config, vocab_src_len, vocab_tgt_len):

    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

import torch
import torch.nn as nn
from model import Transformer
from config import get_config, get_weights_file_path
from train import get_model, get_ds, greedy_decode
import altair as alt
import pandas as pd
import numpy as np
import warnings
from tokenizers import Tokenizer
warnings.filterwarnings("ignore")

import gradio as gr


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config()
train_dataloader, val_dataloader, vocab_src, vocab_tgt  = get_ds(config)
model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)

# load the pretrained wieghts
model_filename = get_weights_file_path(config, f" 4")

# Load the model and map the tensors to the CPU
state = torch.load(model_filename, map_location=torch.device(device))

model.load_state_dict(state["model_state_dict"])

def load_next_batch():
    batch = next(iter(val_dataloader))
    encoder_input = batch['encoder_input'].to(device)
    encoder_mask = batch['encoder_mask'].to(device)
    decoder_input = batch['decoder_input'].to(device)
    decoder_mask = batch['decoder_mask'].to(device)

    encoder_input_tokens = [vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    model_out = greedy_decode(model, encoder_input, encoder_mask, vocab_src, vocab_tgt, config['seq_len'], device)

    return batch, encoder_input_tokens, decoder_input_tokens


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r,c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>" ),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns = ["row", "column", "value", "row_token", "col_token"],
    )


def get_attn_map(attn_type: str, layer: int, head: int):
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention_block.attention_scores

    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention_block.attention_scores

    elif attn_type == "encoder-decoder":  # Cross Attention
        attn = model.encoder.layers[layer].self_attention_block.attention_scores

    return attn[0, head].data


def attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        get_attn_map(attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )

    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(attn_type: str, layers: list[int], heads: list[int], row_tokens:list, col_tokens, max_sentence_len: int):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)

tokenizer_src = Tokenizer.from_file("tokenizer_english_sentence.json") # Getting English Tokenizer file

tokenizer_tgt = Tokenizer.from_file("tokenizer_hindi_sentence.json") # Getting Hindi Tokenizer file


def translation(eng_input):
    max_len = config['seq_len']

    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    enc_input_tokens = tokenizer_src.encode(eng_input).ids
    enc_num_padding_tokens = max_len - len(
        enc_input_tokens) - 2  # Subtracting the two '[EOS]' and '[SOS]' special tokens

    if enc_num_padding_tokens < 0:
        raise ValueError('Sentence is too long')

    # Building the encoder input tensor by combining several elements
    encoder_input = torch.cat(
        [
            sos_token,  # inserting the '[SOS]' token
            torch.tensor(enc_input_tokens, dtype=torch.int64),  # Inserting the tokenized source text
            eos_token,  # Inserting the '[EOS]' token
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)  # Addind padding tokens
        ]
    )

    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()

    model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    return model_out_text


# encoder_input_tokens = tokenizer_src.encode(input_sentence).ids
# decoder_output_tokens = tokenizer_tgt.encode(output_sentence).ids


import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # Dimension of vectors (512)
        self.vocab_size = vocab_size  # Size of the vocabulary

        # Creating a simple lookup table that stores embeddings of a fixed dictionary and size via nn.Embedding
        self.embedding = nn.Embedding(vocab_size,  # size of the dictionary of embeddings
                                      d_model)  # the size of each embedding vector

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Normalizing the variance of the embeddings


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Dimensionality of model
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting

        # Creating a Positional Encoding matrix of shape (seq_len, d_model) filled with zeros
        pe = torch.zeros(seq_len, d_model)

        # Creating a tensor representing positions (0 to seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1)  # Transforming 'position' into a 2D tensor['seq_len', '1']

        # Creating the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the Sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the Cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding an extra dimension at the begining of pe matrix x for batch handling
        pe = pe.unsqueeze(0)  # (1, Seq_Len, d_model)

        # Registering 'pe' as buffer. Buffer is a tensor not considered as a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adding positional encoding to the input tensor X
        # Use x.size(1) to ensure consistent behavior
        # Subtract 1 from x.size(1) to account for 0-based indexing
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(
            False)  ## .requires_grad_() method in PyTorch is used to enable or disable gradient tracking for a tensor
        return self.dropout(x)  # Dropout for regularization


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:  # We define epsilon as 0.00001 to avoid division by zero
        super().__init__()
        self.eps = eps

        # We define alpha as a trainable parameter and initialize it with ones
        self.alpha = nn.Parameter(
            torch.ones(1))  # One-dimensional tensor tensor that will be used to scale the input data

        # We define bias as a trainable parameter and initialize it with zeros
        self.bias = nn.Parameter(torch.zeros(1))  # One-dimensional tensor that will be added to the input data

    def forward(self, x):
        mean = x.mean(dim=-1,
                      keepdim=True)  # Computing the mean of the input data. Keeping the number of dimensions unchanged

        std = x.std(dim=-1,
                    keepdim=True)  # Computing the standard deviation of the input data. Kepping the number of dimensions unchanged

        # Returning the normalized input
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # First linear Transformation
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1

        self.dropout = nn.Dropout(dropout)  # Dropout to prevent overfitting

        # First linear Transformation
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # We ensure that the dimensions of the  model is divisible by the number of heads
        assert d_model % h == 0, "d_model is not divisible by h"

        # d_k is the dimension of ecah attention head's key, query and value vectors
        self.d_k = d_model // h  # d_k formula, like in the originak

        # Defining the weight matrices
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo

        self.dropout = nn.Dropout(dropout)  # Dropout layer to avoid overfitting

    @staticmethod
    def attention(query, key, value, mask,
                  dropout: nn.Dropout):  # mask => When we want certain words to Not interact with others, we "hide" them

        d_k = query.shape[-1]  # The last dimension of query, key and value

        # We Calculate the Attention(Q,K,V) as in the formula
        # (Batch, h, Seq_len, d_k) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # @ = MAtrix multiplication sugn in PyTorch

        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None:  # If a mask is defined...
            attention_scores.masked_fill_(mask == 0, -1e9)  # Replace each value where mask is equal to 0 by -1e9

        # Applying Softmax
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)

        if dropout is not None:  # If a dropout IS defined...
            attention_scores = dropout(attention_scores)  # We apply dropout to prevent overfitting

        # returning self attention and attention score
        return (
                    attention_scores @ value), attention_scores  # Multiply the output Matrix by the V matrix, as in the formula

    def forward(self, q, k, v, mask):

        query = self.w_q(q)  # (Batch, Seq_len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        # Splitting results into smaller matrices for the different heads
        # Splitting embeddings (third dimension) into h parts

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_len, h, d_k) --> ( Batchg, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,
                                                                                       2)  # Transpose => bring the head to the second dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,
                                                                               2)  # Transpose => bring the head to he second dimension
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,
                                                                                       2)  # Transpose => bring the head to the second dimension

        # Obtaining the output and the attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Obtaining the H matrix

        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)  # Multiply the matrix by the weight matrix W_o, resulting in the MH-A matrix


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # We use a dropout layer to prevent overfitting
        self.norm = LayerNormalization()  # We use a normalization layer

    def forward(self, x, sublayer):
        # We normalize the input and add it to the original input 'x'. This craetes the residual connection process.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    # This block takes in the MultiHeadAttentionBlock and FeedForwardBlock, as well as the dropout rate for the residual connections

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()

        # Storing the self-attention block and feed-forward block
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Applying the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x,
                                                                                src_mask))  # Three 'x's corresponding to query, key, and value inputs plus source mask

        # Applying the second residual connection with the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x  # Output tensor after applying self-attention and feed-forward layers with residual connections.


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    # The DecoderBlock takes in two MultiHeadAttentionBlock. One is self-attention, while the other is cross-attention.
    # It also takes in the feed-forward block and the dropout rate
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)])  # List of three Residual Connections with dropout rate

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-Attention block with query, key, and value plus the target language mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # The Cross-Attention block using two 'encoder_ouput's for key and value plus the source language mask. It also takes in 'x' for Decoder queries
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))

        # Feed-forward block with residual connections
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):

    # The Decoder takes in instances of 'DecoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Storing the 'DecoderBlock's
        self.layers = layers
        self.norm = LayerNormalization()  # Layer to normalize the output

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Iterating over each DecoderBlock stored in self.layers
        for layer in self.layers:
            # Applies each DecoderBlock to the input 'x' plus the encoder output and source and target masks
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # Returns normalized output


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:  # Model dimension and the size of the output vocabulary
        super().__init__()
        self.proj = nn.Linear(d_model,
                              vocab_size)  # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_Size)
        return torch.log_softmax(self.proj(x), dim=-1)  # Applying the log Softmax function to the output


class Transformer(nn.Module):

    # This takes in the encoder and decoder, as well the embeddings for the source and target language.
    # It also takes in the Positional Encoding for the source and target language, as well as the projection layer

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Encoder
    def encode(self, src, src_mask):
        src = self.src_embed(src) # Applying source embeddings to the input source language
        src = self.src_pos(src) # Applying source positional encoding to the source embeddings
        return self.encoder(src, src_mask)  # Returning the source embeddings plus a source mask to prevent attention to certain elements

    # Decoder
    def decode(self, encoder_output, src_mask,tgt, tgt_mask):
        tgt = self.tgt_embed(tgt) # Applying target embeddings to the input target language (tgt)
        tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings

        # Returning the target embeddings, the output of the encoder, and both source and target masks
        # The target mask ensures that the model won't 'see' future elements of the sequence
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    # Applying Projection Layer with the Softmax function to the Decoder output
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Function to initialize Transformer

    src_vocab_size: souce language vocabulary size
    tgt_vocab_size: target language vocabulary size
    src_seq_len: source sequence length
    tgt_seq_len: source sequence length
    d_model: Embedding dimension (default = 512)
    N: Number of layers of Encoder and Decoder (default = 6)
    h : Number oh heads for MultiHead Attention (default = 8)
    dropout: Droput
    d_ff: Feed Forward Neural Network's 1st layer dimenion

    return : Transformer Object
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the poistional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []  # Initial list of empty EncoderBlocks
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the Decoder blocks
    decoder_blocks = []  # Initial list of empty DecoderBlocks
    for _ in range(N):  # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)

        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Self-Attention
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Cross-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # FeedForward

        # Combining layers into a DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)  # Appending DecoderBlock to the list of DecoderBlocks

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer  # Assembled and initialized Transformer. Ready to be trained and validated!
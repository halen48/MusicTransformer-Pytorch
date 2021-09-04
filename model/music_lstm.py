import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device


# MusicTransformer
class MusicLSTM(nn.Module):
    """
    ----------
    Author: Guilherme Novaes
    ----------
    """

    def __init__(self, input_size=32, layers=4,
                 hidden_cells = 256):
        super(MusicLSTM, self).__init__()


        self.input_size    = input_size
        self.hidden_cells    = hidden_cells
        self.layers = layers

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.input_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_cells, self.layers)

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.hidden_cells, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x):
        """
        ----------
        Author: Guilherme Novaes
        ----------
        """
        
        x = self.embedding(x)

        x, _ = self.lstm(x)

        y = self.Wout(x)

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # generate
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        ----------
        """

        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END]
            token_probs = y[:, cur_i-1, :]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

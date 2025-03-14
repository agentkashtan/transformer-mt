import os
import sys

import torch

from config import CONFIG
from dataset import c_mask
from transformer import Transformer
from tokenizer import get_tokenizer


def inference(src, model, device, src_len, tgt_len, tokenizer_src, tokenizer_tgt):
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer_src.encode(src).ids
        assert len(src_tokens) - 2 <= src_len
        encoder_input = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('<s>')], dtype=torch.int64),
            torch.tensor(src_tokens, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('</s>')], dtype=torch.int64),
            ])
        src_mask = None
        #src_mask = (encoder_input != tokenizer_src.token_to_id('<pad>')).unsqueeze(0).unsqueeze(0).unsqueeze(0).int()
        # (1, 1, 1, src_len)
        #src_mask = src_mask.to(device)
        # (1, src_len)
        encoder_input = encoder_input.unsqueeze(0).to(device)
        # (1, src_len, d_model)
        encoder_output = model.encode(encoder_input, src_mask)
        #print(encoder_output.shape)        
        # (1, 1)
        decoder_input = torch.tensor([tokenizer_tgt.token_to_id('<s>')], dtype=torch.int64).unsqueeze(0).to(device)
        while decoder_input.size(1) < tgt_len:
            tgt_mask = c_mask(decoder_input.size(-1)).to(device)
            # batch, seq_len, d_model

            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            output = model.project(decoder_output[:, -1, :])
            _, prediction = torch.max(output, dim=-1)
            if prediction.item() == tokenizer_tgt.token_to_id('</s>'):
                break
            decoder_input = torch.cat([decoder_input, prediction.unsqueeze(0)], dim=-1)

        return decoder_input.squeeze(0).cpu()


def main(*args):
    SRC_LEN = CONFIG["src_len"]
    TGT_LEN = CONFIG["tgt_len"]
    VOCAB_SIZE = CONFIG["vocab_size"]
    
    tokenizer_src = get_tokenizer("en")
    tokenizer_tgt = get_tokenizer("fr")

    checkpoint = torch.load(args[0], weights_only=True)
    print(f'epoch: {checkpoint["epoch"]}; loss: {checkpoint["loss"]}')
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, SRC_LEN, TGT_LEN)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    while True: 
        data = input("Enter input sentense: ")
        model_output = inference(data, model, device, SRC_LEN, TGT_LEN, tokenizer_src, tokenizer_tgt)
        print(tokenizer_tgt.decode(model_output.tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main(*sys.argv[1:])

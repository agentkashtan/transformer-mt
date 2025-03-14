import torch
from torch.utils.data import Dataset


class TDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, max_length_src, max_length_tgt) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_length_src = max_length_src
        self.max_length_tgt = max_length_tgt
        self.src_sos_id = self.tokenizer_src.token_to_id("<s>")
        self.src_eos_id = self.tokenizer_src.token_to_id("</s>")
        self.src_pad_id = self.tokenizer_src.token_to_id("<pad>")
        self.tgt_sos_id = self.tokenizer_tgt.token_to_id("<s>")
        self.tgt_eos_id = self.tokenizer_tgt.token_to_id("</s>")
        self.tgt_pad_id = self.tokenizer_tgt.token_to_id("<pad>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        encoder_tokens = self.tokenizer_src.encode(self.dataset[idx]["translation"]["en"]).ids
        decoder_tokens = self.tokenizer_tgt.encode(self.dataset[idx]["translation"]["fr"]).ids

        encoder_padding = self.max_length_src - len(encoder_tokens) - 2
        encoder_input = torch.cat([
            torch.tensor([self.src_sos_id], dtype=torch.int64),
            torch.tensor(encoder_tokens, dtype=torch.int64),
            torch.tensor([self.src_eos_id], dtype=torch.int64),
            torch.tensor([self.src_pad_id] * encoder_padding, dtype=torch.int64)
        ])

        decoder_padding = self.max_length_tgt - len(decoder_tokens) - 1
        decoder_input = torch.cat([
            torch.tensor([self.tgt_sos_id], dtype=torch.int64),
            torch.tensor(decoder_tokens, dtype=torch.int64),
            torch.tensor([self.tgt_pad_id] * decoder_padding, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(decoder_tokens, dtype=torch.int64),
            torch.tensor([self.tgt_eos_id], dtype=torch.int64),
            torch.tensor([self.tgt_pad_id] * decoder_padding, dtype=torch.int64)
        ])

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "src_mask": (encoder_input != self.src_pad_id).unsqueeze(0).unsqueeze(0).int(),
            "tgt_mask": (decoder_input != self.tgt_pad_id).unsqueeze(0).unsqueeze(0).int() & c_mask(self.max_length_tgt),
            "label": label
        }


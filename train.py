import os

from transformer import Transformer
from dataset import TDataset
from tokenizer import get_tokenizer

from datasets import load_dataset
from torch.utils.data import  DataLoader
from tqdm import tqdm


def main():
    books = load_dataset("opus_books", "en-fr", cache_dir="./huggingface_cache")
    en_corcus = [item["translation"]["en"] for item in books["train"]]
    fr_corcus = [item["translation"]["fr"] for item in books["train"]]
    
    VOCAB_SIZE = 30522
    tokenizer_en = get_tokenizer("en", en_corcus)
    tokenizer_en = get_tokenizer("fr", fr_corcus)

    WEIGHTS_PATH = "weights"
    os.makedirs(WEIGHTS_PATH, exist_ok=True)

    #max_length_src + 2
    SRC_LEN = 481
    #max_length_tgt + 2
    TGT_LEN = 596

    data = books["train"].train_test_split(test_size=0.1, shuffle=True)
    training_data = data["train"]
    validation_data = data["test"]
    training_dataset = TDataset(training_data, tokenizer_en, tokenizer_fr, SRC_LEN, TGT_LEN)
    validation_dataset = TDataset(validation_data, tokenizer_en, tokenizer_fr, SRC_LEN, TGT_LEN)
    training_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Devive: {device}')
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, SRC_LEN, TGT_LEN)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_fr.token_to_id("<pad>"), label_smoothing=0.1)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    num_epoch = 30

    step = 0
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch in tqdm(training_dataloader, desc=f'processing epoch: {epoch + 1}'):
            optimizer.zero_grad()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            label = batch["label"].to(device)
            #print(encoder_input.min().item(), encoder_input.max().item())  # Should be within [0, vocab_size-1]
            encoder_output = model.encode(encoder_input, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            prediction = model.project(decoder_output)

            loss = loss_fn(prediction.view(-1, VOCAB_SIZE), label.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            step += 1
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss / max(1, len(training_dataloader))
                }, f'{WEIGHTS_PATH}/model_weights_epoch_{epoch+1}.pt')

        print(f'epoch: {epoch + 1} ------> {total_loss / max(1, len(training_dataloader))}')

if __name__ == "__main__":
    main()


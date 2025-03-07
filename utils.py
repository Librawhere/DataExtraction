import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, filepath, seq_length, tokenizer):
        self.seq_length = seq_length
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
        sentences = self.text.split('\n')
        sentences = " <|endoftext|> ".join(sentences)
        # with open('toy_data/sentences.txt', 'w', encoding='utf-8') as f:
        #     f.write(sentences)
        self.text_idx = tokenizer(sentences).data['input_ids']
        self.length = self.__len__()

    def __len__(self):
        return len(self.text_idx) - self.seq_length

    def __getitem__(self, idx):
        inputs = torch.tensor(self.text_idx[idx:idx + self.seq_length])
        targets = torch.tensor(self.text_idx[idx + 1:idx + self.seq_length + 1])
        return inputs, targets


def train(config, model, train_loader, optimizer, scheduler, device, tokenizer, logger):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    lss = []
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, targets)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, config['num_epochs'],
                                                                         i + 1, len(train_loader), loss.item()))
        lss.append(total_loss / len(train_loader))
        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, config['num_epochs'], total_loss / len(train_loader)))

        model.eval()
        logger.info('------------------')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'weights/adatr{epoch+1}.pt')

    return lss





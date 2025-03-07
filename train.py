import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import TextDataset, train
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from model import Pythia
from torch.utils.data import DataLoader
from datetime import datetime


def pythia_portal():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    t = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    fh = logging.FileHandler(f'logs/log_{t}.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'seq_length': 120,
        'batch_size': 128,
        'learning_rate': 2e-5,
        'num_epochs': 70,
        'weight_decay': 0.01,
        'clip': 1.0,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )

    train_dataset = TextDataset('dataset/data.json', config['seq_length'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = Pythia().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  eps=1e-8,
                                  weight_decay=config['weight_decay'])
    total_steps = (train_dataset.length // config['batch_size']) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    lss = train(config, model, train_loader, optimizer, scheduler, device, tokenizer, logger)

    plt.title('loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(config['num_epochs']), lss, color='r')
    pth = './results/{}.jpg'.format('loss')
    plt.savefig(pth)
    plt.close()


if __name__ == '__main__':
    pythia_portal()

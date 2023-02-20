import torch
import random
from torchtext.datasets import IMDB, CoLA
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
MAX_VOCAB = 25004

"""
New torchtext:
https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb
https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71
"""

"""
Fetches batches for either IMDB or CoLA dataset
"""
def fetch_IMDB(batch_size=64):
    # Process datasets
    train, test = IMDB(split=('train', 'test'))
    # Split train data by 70%
    train, valid = train.random_split(total_length=len(list(train)),
                                  weights={"train": 0.7, "valid": 0.3},
                                  seed=SEED)
    # Vocab
    tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train),
                                      specials=['<UNK>', '<PAD>', '<BOS>', '<EOS>'],
                                      max_tokens=MAX_VOCAB)
    vocab.set_default_index(vocab['<UNK>'])

    text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]

    train_list  = list(train)
    valid_list = list(valid)
    test_list = list(test)

    """
    Helper functions to generate batches
    """
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
                label_list.append(_label)
                processed_text = torch.tensor(text_transform(_text))
                text_list.append(processed_text)
        return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

    def batch_sampler(data_list):
        indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(data_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]

    train_iterator = DataLoader(train_list, batch_sampler=batch_sampler(train_list),
                                collate_fn=collate_batch)
    valid_iterator = DataLoader(valid_list, batch_sampler=batch_sampler(valid_list),
                                collate_fn=collate_batch)
    test_iterator = DataLoader(test_list, batch_sampler=batch_sampler(test_list),
                                collate_fn=collate_batch)
    return train_iterator, valid_iterator, test_iterator

"""
Fetches batches for either IMDB or CoLA dataset
"""
def fetch_CoLA(batch_size=64):
    # Process datasets
    train, valid, test = CoLA(split=('train', 'dev', 'test'))
    # Vocab
    tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
    def yield_tokens(data_iter):
        for _, _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train),
                                      specials=['<UNK>', '<PAD>', '<BOS>', '<EOS>'],
                                      max_tokens=MAX_VOCAB)
    vocab.set_default_index(vocab['<UNK>'])

    text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]

    train_list  = list(train)
    valid_list = list(valid)
    test_list = list(test)

    """
    Helper functions to generate batches
    """
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_, _label, _text) in batch:
                label_list.append(_label)
                processed_text = torch.tensor(text_transform(_text))
                text_list.append(processed_text)
        return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

    def batch_sampler(data_list):
        indices = [(i, len(tokenizer(s[2]))) for i, s in enumerate(data_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]

    train_iterator = DataLoader(train_list, batch_sampler=batch_sampler(train_list),
                                collate_fn=collate_batch)
    valid_iterator = DataLoader(valid_list, batch_sampler=batch_sampler(valid_list),
                                collate_fn=collate_batch)
    test_iterator = DataLoader(test_list, batch_sampler=batch_sampler(test_list),
                                collate_fn=collate_batch)
    return train_iterator, valid_iterator, test_iterator

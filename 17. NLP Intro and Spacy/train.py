import torch
from torch.autograd import Variable
from torch.nn import NLLLoss
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from clean_data import *


preprocessed = preprocess('17. NLP Intro and Spacy/data/article.txt')


cleaned = clean_sentences(preprocessed)


word_idx, vocab_size = get_dicts(cleaned)

pairs = get_pairs(cleaned, word_idx, 2)


def get_embedding(embeddings, word_to_idx, word):
    word = word.lower()
    idx = word_to_idx[word]
    return embeddings[:, idx]


def input_layer(word_idx, vocab_size):
    x = torch.zeros(vocab_size)
    x[word_idx] = 1.0
    return x


def train(dataset, n_epochs, print_every, lr, embedding_size, vocab_size):
    criterion = NLLLoss()
    W1 = Variable(torch.randn(
        vocab_size, embedding_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(embedding_size,
                  vocab_size).float(), requires_grad=True)

    # x = Variable(input_layer(dataset[0][0], vocab_size)).float()
    # print(x.shape)
    # y_true = Variable(torch.tensor([dataset[0][1]])).long()
    # print(y_true.type())

    losses = []
    for epoch in range(n_epochs):

        loss_val = 0
        running_loss = 0

        for data, target in dataset:

            x = Variable(input_layer(data, vocab_size)).float()
            y_true = Variable(torch.tensor([target])).long()

            z1 = torch.matmul(x, W1)
            z2 = torch.matmul(z1, W2)

            log_softmax = F.log_softmax(z2, dim=0)

            loss = criterion(log_softmax.view(1, -1),  y_true)
            loss.backward()

            W1.data -= lr * W1.grad
            W2.data -= lr * W2.grad

            W1.grad.zero_()
            W2.grad.zero_()

            loss_val += loss.item()
            running_loss += loss.item()

        if epoch % print_every == 0 and len(losses) != 0:
            print(f'Loss at epoch {epoch}: {losses[-1]}')

        losses.append(loss_val/len(dataset))

    return W2.detach(), losses


embeddings, losses = train(pairs, 40, 5,  0.01, 10, vocab_size)

plt.plot(losses)
plt.show()


print(embeddings.shape)
print(get_embedding(embeddings, word_idx, 'Saudi'))
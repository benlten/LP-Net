import torch
import torchvision
import tqdm
import sys

from sklearn.linear_model import LogisticRegression

def pprint(t):
    print('\n'.join([''.join([(' ' if i == 0 else '*') for i in l]) for l in t.mean(dim=0)]))

def loss(Z, K, margin = 100):
    idx_range = torch.arange(len(Z)).reshape(-1,1) // K * K
    pos_indexes = idx_range + torch.arange(K)
    pos_tensors = Z[pos_indexes]
    pos_distance = (Z.unsqueeze(1) - pos_tensors).norm(dim=-1)
    corrected_pos_distance = 0.5 * (pos_distance**2)
    pos_similarity = torch.nn.functional.cosine_similarity(Z.unsqueeze(1), pos_tensors, dim=-1)
    corrected_pos_similarity = 0.5 * ((1-pos_similarity)**2) 

    neg_indexes = torch.randint(0, len(Z)-K, (len(Z),K))
    # all_neg_indexes = torch.arange(0, len(Z) - K).repeat(len(Z),1)
    neg_indexes_corrected = (neg_indexes + idx_range + K) % len(Z)
    neg_tensors = Z[neg_indexes_corrected]
    neg_distance = (Z.unsqueeze(1) - neg_tensors).norm(dim=-1)
    corrected_neg_distance = 0.5*(torch.nn.functional.relu(margin-neg_distance) ** 2)
    neg_similarity = torch.nn.functional.cosine_similarity(Z.unsqueeze(1), neg_tensors, dim=-1)
    corrected_neg_similarity = 0.5 * (neg_similarity**2) 

    return corrected_neg_similarity.mean() + corrected_pos_similarity.mean(), neg_similarity.mean(), pos_similarity.mean()
    return corrected_neg_distance.mean() + corrected_pos_distance.mean(), neg_distance.mean(), pos_distance.mean()

N = 10
K = 512
D = 32

if False:
    MNIST = torchvision.datasets.MNIST('.', download=True)
    X = torch.stack([
        MNIST.data[MNIST.targets == n][:K]
        for n in range(N)
      ]).unsqueeze(2).float()
else: 
    CIFAR = torchvision.datasets.CIFAR10('.', download=True)
    X = torch.stack([
        torch.tensor(CIFAR.data)[torch.tensor(CIFAR.targets) == n][:K]
        for n in range(N)
      ]).float().transpose(-1,-2).transpose(-2,-3)
    print(X.shape)

y = torch.arange(N).reshape(-1,1).repeat(1,K)
print(y.shape)

_,_,C,H,W = X.shape

print(X.numel() * X.element_size())
X = X.flatten(0,1).cuda()
y = y.flatten(0,1).cuda()

print(X.flatten(1).shape)

l = LogisticRegression().fit(
        X.flatten(1).detach().cpu().numpy(), 
        y.detach().cpu().numpy()
)
y_hat = torch.tensor(l.predict_log_proba(X.flatten(1).cpu().detach())).cuda()

print('raw')
print(torch.nn.functional.cross_entropy(y_hat, y.long()))
print((y_hat.argmax(dim=1) == y.long()).float().mean())


model = torch.nn.Sequential(
    torch.nn.Conv2d(C, D, 7),
    torch.nn.ReLU(),
    torch.nn.Conv2d(D, D, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(D, D, 3),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(D, D)
  ).cuda()

optimizer = torch.optim.Adam(model.parameters())

l = LogisticRegression().fit(
        model(X).detach().cpu().numpy(), 
        y.detach().cpu().numpy()
)

y_hat = torch.tensor(l.predict_log_proba(model(X).cpu().detach())).cuda()

print('before')
print(torch.nn.functional.cross_entropy(y_hat, y.long()))
print((y_hat.argmax(dim=1) == y.long()).float().mean())


with tqdm.trange(10000) as pbar:
    for _ in (pbar): 
        Z = model(X)
        l, neg, pos = loss(Z, K)
        pbar.set_description(f'loss: {l**0.5}, neg: {neg}, pos: {pos}')
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


l = LogisticRegression().fit(
        model(X).detach().cpu().numpy(), 
        y.detach().cpu().numpy()
)

y_hat = torch.tensor(l.predict_log_proba(model(X).cpu().detach())).cuda()

print('after')
print(torch.nn.functional.cross_entropy(y_hat, y.long()))
print((y_hat.argmax(dim=1) == y.long()).float().mean())

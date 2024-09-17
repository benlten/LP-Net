import torch
from transformations import (
        SalienceSamplingOld as SalienceSampling, 
        LogPolar, 
        NRandomCrop, 
        Compose, 
        Resize, 
        Foveation, 
        FoveationOld, 
        Identity
)
import torchvision
import timeit

N = 1000

def warmup(device = 'cuda', N = 1000):
    for _ in range(N):
        torch.randn(1000,1000).to(device)

print('Foveation')
s = Foveation(180)
print('cpu tensor, move to cuda', timeit.timeit(lambda: s(torch.randn(10,3,180,180)).cuda(), number=N // 100))
print('cuda tensor', timeit.timeit(lambda: s(torch.randn(10,3,180,180).cuda()), number=N // 100))

# print('warming cuda...', end='\r')
# warmup('cuda', N)
# print('cuda warmup completed')
# print('warming cpu...', end='\r')
# warmup('cpu', N)
# print('cpu warmup completed')
# 
# print('LogPolar')
# lp1 = LogPolar(input_shape = (180,180), output_shape = (190,165))
# lp2 = LogPolar(input_shape = (180,180), output_shape = (190,165)).cuda()
# print('cpu op, cpu tensor, explicit move to cuda', timeit.timeit(lambda: lp1(torch.randn(1,3,180,180)).cuda(), number=N))
# print('cuda module, cuda tensor', timeit.timeit(lambda: lp2(torch.randn(1,3,180,180).cuda()), number=N))
# 
# 
# print('RandomRotation')
# R = torchvision.transforms.RandomRotation(90).cuda()
# print('cpu tensor, explicit move to cuda', timeit.timeit(lambda: R(torch.randn(3,180,180)).cuda(), number=N))
# print('cuda tensor', timeit.timeit(lambda: R(torch.randn(3,180,180).cuda()), number=N))
# 
# 
# print('Transformations')
# for T in [ NRandomCrop(4,180), Resize(4, 180) ]:
#     print(T.__class__)
#     print('cpu tensor, explicit move to cuda', timeit.timeit(lambda: T(torch.randn(3,224,224)).cuda(), number=N))
#     print('cuda tensor', timeit.timeit(lambda: T(torch.randn(3,224,224).cuda()), number=N))

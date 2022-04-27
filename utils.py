import torch
from functools import reduce
from model.RSlayer import ParabolicIntegrate
from torch.utils.data import TensorDataset, DataLoader

def cacheXiFeature(graph, T, X, W, device, batch_size = 100):
    '''
    return features only containing Xi
    '''
    InteLayer = ParabolicIntegrate(graph, T = T, X = X).to(device)
    WSet = TensorDataset(W)
    WLoader = DataLoader(WSet, batch_size=batch_size, shuffle=False)
    XiFeature = []
    for (W, ) in WLoader:
        XiFeature.append(InteLayer(W = W.to(device)).to('cpu'))
    XiFeature = torch.cat(XiFeature, dim = 0)
    return XiFeature

# use torch.profiler.profile to get the time of each layer
def profile(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes = False,
            with_modules = True,
            with_stack = True,
            profile_memory=False,
            with_flops = False) \
        as prof:
            output = model(U0, W, F_Xi)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        break

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

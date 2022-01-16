from torch.autograd import Variable


def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        var = Variable(tensor)
    else:
        var = Variable(tensor, requires_grad=requires_grad)
    return var

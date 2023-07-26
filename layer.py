import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from utils import sparse_dropout, dot


# Binarized to '+1' and '-1'
class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


# Binarized to '+1' and '0'
class BinaryQuantizeZ(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out.lt(0)] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class BiLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, binary_act=True, alpha=1.0):
        super(BiLinear, self).__init__(in_features, out_features, bias=False)
        self.binary_act = binary_act
        self.output_ = None
        self.have_bias = bias

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = BinaryQuantize().apply(bw)
        if self.binary_act:
            ba = BinaryQuantizeZ().apply(ba)
        #         print('BW is',bw)
        #         print('BA is',ba)
        if self.have_bias == True:
            output = F.linear(ba, bw, self.bias)
        else:
            #             print('dose not have bias')
            output = F.linear(ba, bw)
        self.output_ = output
        return output


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        # self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        # self.weight = nn.Linear(input_dim, output_dim, bias=False)
        self.weight = BiLinear(input_dim, output_dim, bias=False)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs
        x, support = x.to_dense(), support.to_dense()

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                # xw = torch.sparse.mm(x, self.weight)
                xw = self.weight(x)
            else:
                # xw = torch.mm(x, self.weight)
                xw = self.weight(x)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution

from    config import args

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        self.conv1 = GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False)

        self.conv2 = GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False)

        
        # self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
        #                                              activation=F.relu,
        #                                              dropout=args.dropout,
        #                                              is_sparse_inputs=True),

        #                             GraphConvolution(args.hidden, output_dim, num_features_nonzero,
        #                                              activation=F.relu,
        #                                              dropout=args.dropout,
        #                                              is_sparse_inputs=False),

        #                             )

    def forward(self, inputs):
        x, support = inputs

        # x = self.layers((x, support))
        x, support = self.conv1((x, support))
        x = self.conv2((x, support))

        return x

    def l2_loss(self):

        # layer = self.layers.children()
        # layer = next(iter(layer))

        layer = self.conv2
        loss = None

        for name, p in layer.named_parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss

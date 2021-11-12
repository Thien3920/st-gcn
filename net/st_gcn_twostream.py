import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN



class Model(nn.Module):
    """Two inputs spatial temporal graph convolutional networks.
    Args:
        - graph_args: (dict) Args map of `Actionsrecognition.Utils.Graph` Class.
        - num_class: (int) Number of class outputs.
        - edge_importance_weighting: (bool) If `True`, adds a learnable importance
            weighting to the edges of the graph.
        - **kwargs: (optional) Other parameters for graph convolution units.
    Shape:
        - Input: :tuple of math:`((N, 3, T, V), (N, 2, T, V))`
        for points and motions stream where.
            :math:`N` is a batch size,
            :math:`in_channels` is data channels (3 is (x, y, score)), (2 is (mot_x, mot_y))
            :math:`T` is a length of input sequence,
            :math:`V` is the number of graph nodes,
        - Output: :math:`(N, num_class)`
    """
    def __init__(self, graph_args, num_class, edge_importance_weighting=True,
                 **kwargs):
        super().__init__()
        self.pts_stream = ST_GCN(3, graph_args, None,
                                                     edge_importance_weighting,
                                                     **kwargs)
        self.mot_stream = ST_GCN(2, graph_args, None,
                                                     edge_importance_weighting,
                                                     **kwargs)

        self.fcn = nn.Linear(256 * 2, num_class)

    def forward(self, inputs):
        out1 = self.pts_stream(inputs[0])
        out2 = self.mot_stream(inputs[1])

        concat = torch.cat([out1, out2], dim=-1)
        out = self.fcn(concat)

        return torch.sigmoid(out)

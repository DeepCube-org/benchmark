import torch
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.conv import GATv2Conv, GINEConv, GENConv


class UniMPLayer(torch.nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            act,
            dropout,
            heads, 
            concat,
        ):
        super(UniMPLayer, self).__init__()
        self.act = act
        self.norm = torch.nn.LayerNorm(in_channels*heads, elementwise_affine=True)
        self.conv_h = TransformerConv(
            in_channels=in_channels, 
            out_channels=out_channels,
            beta = True, 
            heads = heads, 
            concat = concat
        )
        self.droput = torch.nn.Dropout(dropout)
        
        
    def forward(self, x, edge_index):
        x = self.conv_h(x, edge_index)
        x = self.norm(x)
        if(self.act is not None):
            x = self.act(x)
        x = self.droput(x)
        return(x)

class GATv2Layer(torch.nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            act,
            dropout,
            heads, 
            concat,
        ):
        super(GATv2Layer, self).__init__()
        self.act = act
        self.norm = torch.nn.LayerNorm(in_channels*heads, elementwise_affine=True)
        self.conv_h = GATv2Conv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            heads = heads, 
            concat = concat,
            add_self_loops = True
        )
        self.droput = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv_h(x, edge_index)
        x = self.norm(x)
        if(self.act is not None):
            x = self.act(x)
        x = self.droput(x)
        return(x)


class GINELayer(torch.nn.Module):
    def __init__(self, 
            in_channels: int, 
            out_channels: int,
            act,
            dropout,
            heads  = None, # Not used
            concat = None, # Not used
        ):
        super(GINELayer, self).__init__()
        self.act = act
        self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=True)
        self.conv_h = GINEConv(
            nn = torch.nn.Sequential(
               torch.nn.Linear(in_features=in_channels, out_features=out_channels),
            ), 
            train_eps=True
        )
        self.droput = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        h = self.norm(h)
        if(self.act is not None):
            h = self.act(h)
        h = self.conv_h(h, edge_index)
        x = h+x
        x = self.droput(x)
        return(x)


class DyResLayer(torch.nn.Module):
    """
    Implementation of DeeperConv
    """
    def __init__(self, 
            in_channels: int, 
            out_channels: int,
            act,
            dropout,
            heads  = None, # Not used
            concat = None, # Not used
        ):
        super(DyResLayer, self).__init__()
        self.act = act
        self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=True)
        self.conv_h = GENConv(
            in_channels = in_channels,
            out_channels =  out_channels,
            num_layers = 2,
            norm = 'layer',
            aggr='softmax',
            t=1.0, 
            learn_t=True,
            msg_norm = True,
            learn_msg_scale = True
        )
        self.droput = torch.nn.Dropout(dropout)
    def forward(self, x, edge_index):
        h = x
        h = self.norm(h)
        if(self.act is not None):
            h = self.act(h)
        h = self.conv_h(h, edge_index)
        x = h+x
        x = self.droput(x)
        return(x)

class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers, 
        layer,
        dropout,
        heads,
    ):
        super(GraphEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.hidden_graph_layers = torch.nn.ModuleList()
        assert self.n_layers>0, 'The minimum number of layers is 1'


        self.hidden_graph_layers.append(
            layer(
                in_channels = hidden_dim, 
                out_channels = hidden_dim,
                act = torch.nn.ReLU(),  
                heads=heads, 
                concat=self.n_layers>1,
                dropout=dropout
            )
        )
        
        for i in range(1, self.n_layers):
            
            concat = i<(self.n_layers-1) # if it is the last layer, the output must not be concatenated 

            hidden_graph_layer = layer(
                in_channels = hidden_dim*heads, 
                out_channels = hidden_dim,
                act = torch.nn.ReLU(),  
                heads=heads, 
                concat=concat,
                dropout=dropout
            )
            self.hidden_graph_layers.append(hidden_graph_layer)
        


    def forward(self, x, edge_index):
        hidden = x
        for i in range(len(self.hidden_graph_layers)):
            hidden = self.hidden_graph_layers[i](x = hidden, edge_index = edge_index)
        return(hidden)

class GraphModel(torch.nn.Module):

    LAYER_TYPES = {
        'DyResLayer': DyResLayer,
        'UniMPLayer': UniMPLayer,
        'GINELayer': GINELayer,
        'GATv2Layer': GATv2Layer
    }

    def __init__(
        self, 
        layer,
        n_layers, 
        hidden_dim, 
        dropout,
        heads
    ):
        super(GraphModel, self).__init__()
        self.graph_encoder = GraphEncoder(
            hidden_dim = hidden_dim, 
            n_layers = n_layers, 
            heads = heads, 
            layer = layer, 
            dropout = dropout
        )

    def forward(self, x, edge_index):
        hidden = self.graph_encoder(x, edge_index)
        return(hidden)
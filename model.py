from config import *
from utils import *


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()
    
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels, heads=8, dropout=0.0, edge_dim=edge_dim
        )
        self.conv2 = TransformerConv(
            out_channels * 8,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.0,
            edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)       
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)

        h = self.lin_seq(h)

        return h


class SparseAttentionTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, device):
        super(SparseAttentionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4, 
                activation='relu',
                batch_first = True),
            num_layers=6
        )

    def forward(self, x, mask):
        # 将节点输入映射到嵌入空间
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        # 应用mask
        x = self.transformer_encoder(x, mask=mask)
        # print(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.sep_embedding = nn.Embedding(1, d_model)
        self.dropout = nn.Dropout(dropout)
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)  # 自注意力层权重
            nn.init.xavier_uniform_(layer.linear1.weight)   

    def forward(self, src1, src2, mask1, mask2):
        # x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # Scale embeddings
        # src1.shape = (batch, padding_len, 100)
        # src2.shape = (batch, padding_len, 100)
        # mask1.shape = (batch, padding_len)
        # mask2.shape = (batch, padding_len)
        sep = self.sep_embedding(torch.tensor([0], device=src1.device)).repeat(src1.shape[0], 1, 1)
        # sep.shape = (batch, 1, 100)
        sep_mask = torch.zeros(src1.shape[:1], dtype=torch.long, device=mask1.device).reshape(-1, 1)
        # sep_mask.shape = (batch, 1)
        x = torch.cat((src1, sep, src2), dim=1)
        mask = torch.cat([mask1, sep_mask, mask2], dim=1)
        
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x

class TransformerEncoder2(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder2, self).__init__()
        # self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.sep_embedding = nn.Embedding(1, d_model)
        self.dropout = nn.Dropout(dropout)
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)  # 自注意力层权重
            nn.init.xavier_uniform_(layer.linear1.weight)   

    def forward(self, src, mask, sep_len):
        # x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # Scale embeddings
        # src.shape = (n_id, 100)
        # sep.shape = (batch_size, 100)
        sep = self.sep_embedding(torch.tensor([0], device=src.device)).repeat(sep_len, 1)
        # sep.shape = (batch, 1, 100)
        # sep_mask.shape = (batch, 1)
        x = torch.cat((src, sep), dim=0).unsqueeze(0)
        mask = torch.tensor(mask, device=src.device, dtype=float)
        
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        return x

class EdgePredictor(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(EdgePredictor, self).__init__()
        self.lin_seq = nn.Sequential(
            Linear(in_channels, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),
        )
        # nn.init.xavier_uniform_(self.lin_seq.weight)
    
    def forward(self,x):
        x = self.lin_seq(x)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)

        h = self.lin_seq(h)

        return h



class RandomNeighborLoader(object):
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long,
                                     device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long,
                                device=device)
        self.__assoc__ = torch.empty(num_nodes, dtype=torch.long,
                                     device=device)

        self.reset_state()

    def __call__(self, n_id):
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
        self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self.__assoc__[neighbors], self.__assoc__[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src, dst):
        # Inserts newly encountered interactions into an ever growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self.__assoc__[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        # Randomly select indices instead of using the first `size` elements
        perm = torch.randperm(nodes.size(0), device=nodes.device)
        selected = perm[:min(nodes.size(0), self.size)]
        
        dense_id = torch.arange(selected.size(0), device=nodes.device) % self.size
        dense_id += self.__assoc__[nodes[selected]].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size, ), -1)
        dense_e_id[dense_id] = e_id[selected]
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors[selected]
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, :self.size], dense_neighbors], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)


class MoE_GNN_Transformer(nn.Module):
    def __init__(self, gnn, transformer, edge_pred, hidden_size, device):
        super(MoE_GNN_Transformer, self).__init__()
        self.gnn = gnn
        self.transformer = transformer
        self.edge_pred = edge_pred
        
        # 门控网络，用于选择专家（这里是一个简单的MLP）
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # 输出两个值，一个表示一跳专家，另一个表示两跳专家的概率
        )
        
        # 隐藏层的维度
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, src, pos_dst, edge_index, assoc, train_data, batch_size):
        # 通过GNN计算节点的表示
        z, last_update = self.gnn(src, pos_dst, edge_index, train_data)
        
        # 将输出传递给门控网络
        gate_output = self.gate_network(z)
        gate_probs = F.softmax(gate_output, dim=-1)  # softmax保证概率值在0-1之间
        
        # 使用门控网络的输出来选择一跳专家和两跳专家
        onehop_features = self.onehop_expert(src, pos_dst, assoc, z)
        twohop_features = self.twohop_expert(src, pos_dst, edge_index, assoc, z)
        
        # 计算专家的加权输出
        onehop_output = onehop_features * gate_probs[:, 0].unsqueeze(-1)
        twohop_output = twohop_features * gate_probs[:, 1].unsqueeze(-1)
        
        # 组合一跳和两跳专家的输出
        output = onehop_output + twohop_output
        
        # 最终通过Transformer和边预测网络
        transformer_output = self.transformer(output)
        sep_feature = transformer_output[:, -1, :]  # 使用Transformer输出的sep
        pos_out = self.edge_pred(sep_feature)
        
        return pos_out

    def onehop_expert(self, src, pos_dst, assoc, z):
        # 处理一跳邻居信息
        src_features = z[assoc[src]]
        dst_features = z[assoc[pos_dst]]
        
        # 填充并返回一跳特征
        src_feature_seq = nn.utils.rnn.pad_sequence(src_features, batch_first=True)
        dst_feature_seq = nn.utils.rnn.pad_sequence(dst_features, batch_first=True)
        
        return src_feature_seq, dst_feature_seq

    def twohop_expert(self, src, pos_dst, edge_index, assoc, z):
        # 处理两跳邻居信息
        src_features, dst_features = [], []
        
        for src1, dst1 in zip(src, pos_dst):
            neighbor1 = edge_index[1][edge_index[0] == assoc[src1]].unique()
            neighbor2 = edge_index[1][edge_index[0] == assoc[dst1]].unique()

            tmp1 = torch.cat([assoc[src1].unsqueeze(0), neighbor1], dim=-1)
            tmp2 = torch.cat([assoc[dst1].unsqueeze(0), neighbor2], dim=-1)

            neighbor1_feature = z[tmp1]
            neighbor2_feature = z[tmp2]

            src_features.append(neighbor1_feature)
            dst_features.append(neighbor2_feature)
        
        # 填充并返回两跳特征
        src_feature_seq = nn.utils.rnn.pad_sequence(src_features, batch_first=True)
        dst_feature_seq = nn.utils.rnn.pad_sequence(dst_features, batch_first=True)
        
        return src_feature_seq, dst_feature_seq

# 定义 MoE 模块
class MoE(nn.Module):
    def __init__(self, input_size=100, num_experts=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.fc = nn.Linear(input_size, num_experts)

    def forward(self, x):
        # 根据输入x计算一个选择，选择输出1跳或2跳
        logits = self.fc(x)
        # 使用 softmax 选择跳数，得出每个专家的概率分布
        probs = F.softmax(logits, dim=-1)
        # 使用 Gumbel-Softmax 或直接取最大概率来选择跳数
        chosen_expert = torch.argmax(probs, dim=-1)

        return chosen_expert, probs
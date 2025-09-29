# encoding=utf-8
import logging

from config import *
from model import *

logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def cal_pos_edges_loss(link_pred_ratio):
    loss = []
    for i in link_pred_ratio:
        loss.append(criterion(i, torch.ones(1)))
    return torch.tensor(loss)


def cal_pos_edges_loss_multiclass(link_pred_ratio, labels):
    loss = []
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1, -1), labels[i].reshape(-1)))
    return torch.tensor(loss)


def cal_pos_edges_loss_autoencoder(decoded, msg):
    loss = []
    for i in range(len(decoded)):
        loss.append(criterion(decoded[i].reshape(1, -1), msg[i].reshape(-1)))
    return torch.tensor(loss)


connect = psycopg2.connect(
    database="tc_e5_clearscope_dataset_db",
    host="localhost",
    user="postgres",
    password="123456",
    port="5432",
)

cur = connect.cursor()


# train_data = graph_5_8

# # Constructing the map for nodeid to msg
# sql = "select * from node2id ORDER BY index_id;"
# cur.execute(sql)
# rows = cur.fetchall()

# nodeid2msg = {}  # nodeid => msg and node hash => nodeid
# for i in rows:
#     nodeid2msg[i[0]] = i[-1]
#     nodeid2msg[i[-1]] = {i[1]: i[2]}

# rel2id = {
#     1: "EVENT_ACCEPT",
#     "EVENT_ACCEPT": 1,
#     2: "EVENT_CLONE",
#     "EVENT_CLONE": 2,
#     3: "EVENT_CLOSE",
#     "EVENT_CLOSE": 3,
#     4: "EVENT_CREATE_OBJECT",
#     "EVENT_CREATE_OBJECT": 4,
#     5: "EVENT_EXECUTE",
#     "EVENT_EXECUTE": 5,
#     6: "EVENT_OPEN",
#     "EVENT_OPEN": 6,
#     7: "EVENT_READ",
#     "EVENT_READ": 7,
#     8: "EVENT_RECVFROM",
#     "EVENT_RECVFROM": 8,
#     9: "EVENT_SENDTO",
#     "EVENT_SENDTO": 9,
#     10: "EVENT_WRITE",
#     "EVENT_WRITE": 10,
# }

# train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
# max_node_num = max(torch.cat([data.dst,data.src]))+1
# max_node_num = data.num_nodes+1
max_node_num = 139961  # +1
# min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
min_dst_idx, max_dst_idx = 0, max_node_num
# neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)


class getComponent:
    def __init__(self, n_id, edge_index, batch):
        self.data = Data(x=n_id, edge_index=edge_index)
        # print("节点 ", data.x)
        # print(f"src{src}, dst{pos_dst}")
        # print("原始边 ", data.edge_index)
        undirected_edge_index = self.directed_to_undirected(self.data.edge_index)
        self.data.edge_index = undirected_edge_index

        self.batch = batch
        self.base_folder = "./dfs_visual_test"

    # 2. 定义DFS来查找连通域
    def dfs(self, graph, node, visited, component):
        visited[node] = True
        component.append(node)
        neighbors = list(graph.neighbors(node))

        for neighbor in neighbors:
            if not visited[neighbor]:
                self.dfs(graph, neighbor, visited, component)

    def find_connected_components(self, graph):
        visited = [False] * graph.number_of_nodes()
        connected_components = []

        for node in range(graph.number_of_nodes()):
            if not visited[node]:
                component = []
                self.dfs(graph, node, visited, component)
                connected_components.append(component)

        return connected_components

    def directed_to_undirected(self, edge_index):
        row, col = edge_index
        # 将边反向
        reversed_edges = torch.stack([col, row], dim=0)
        # 合并原有边和反向边
        undirected_edge_index = torch.cat([edge_index, reversed_edges], dim=1)
        return undirected_edge_index

    def visualize(self):
        G = to_networkx(self.data, to_undirected=True)
        connected_components = self.find_connected_components(G)
        pos = nx.spring_layout(G)  # 为图的布局生成位置

        for i, component in enumerate(connected_components):
            component_folder = os.path.join(self.base_folder, str(self.batch))
            if not os.path.exists(component_folder):
                os.makedirs(component_folder)
            plt.figure()
            subgraph = G.subgraph(component)
            labels = {node: str(node) for node in subgraph.nodes()}
            nx.draw(
                subgraph,
                pos,
                labels=labels,
                with_labels=True,
                node_color=f"C{i}",
                node_size=700,
                font_size=16,
            )
            plt.title(f"连通域 {i + 1}")
            file_path = os.path.join(component_folder, f"component_{i + 1}.png")
            plt.savefig(file_path)
            plt.close()  # 关闭当前绘制窗口，避免占用内存
            plt.show()
        log_file = os.path.join(component_folder, f"batch{self.batch}.log")
        with open(log_file, "w") as f:
            f.write(f"节点{self.data.x}\n")
            f.write(f"边{self.data.edge_index}\n")
            f.write(f"连通域{connected_components}\n")


# Helper vector to map global node indices to local ones.


def train(
    train_data,
    memory,
    gnn,
    link_pred,
    neighbor_loader,
    optimizer,
    criterion,
    assoc,
    BATCH,
    device
):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes = set()

    total_loss = 0
    total_accuracy = 0
    for batch in train_data.seq_batches(batch_size=BATCH):
        # batch:
        # TemporalData(dst=[1024], msg=[1024, 42], src=[1024], t=[1024])
        # 按照t的顺序取batch
        optimizer.zero_grad()
        src, pos_dst, t, msg, tag = batch.src, batch.dst, batch.t, batch.msg, batch.tag
        n_id = torch.cat([src, pos_dst]).unique()

        n_id, edge_index, e_id = neighbor_loader(n_id)
    
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        
        # edge_num += edge_index.shape[1]
        # cp = getComponent(n_id, edge_index, idx)
        # cp.visualize()

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        # z.size = (n_id, 100) n_id个节点的表示
        # assoc: 将对应位置上的nid按顺序
        # 例如，assoc长为15，nid=[2,4,6,8,10]，则将assoc[2,4,6,8,10]的位置置为[0,1,2,3,4]

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        y_pred = torch.cat([pos_out], dim=0)

        y_true = []
        # for m in msg:
        #     l = tensor_find(m[16:-16], 1) - 1
        #     y_true.append(l)

        y_true = torch.tensor(tag).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        y_pred_acc = torch.argmax(y_pred, dim=1)
        
        accuracy = torch.sum(y_true ==  y_pred_acc).item()
        

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        

        total_loss += float(loss) * batch.num_events
        total_accuracy += float(accuracy)

    return total_loss / train_data.num_events, total_accuracy / train_data.num_events

if __name__ == "__main__":
    train_graphs = torch.load('/mnt/hdd.data/hzc/ATLAS/paper_experiments/S4/training_logs/S1-CVE-2015-5122_windows/train_graphs.simple').to(device)
    train_data = train_graphs[-1]
    print(device)
    print(train_graphs)
    print(artifact_dir)


    # min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    min_dst_idx, max_dst_idx = 0, max_node_num

    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
    
    
    print(max_node_num)
    neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)

    memory = TGNMemory(
        max_node_num,
        train_data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(train_data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=train_data.msg.size(-1),
        time_enc=TimeEncoder(time_dim),
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim, out_channels=2).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=0.00005,
        eps=1e-08,
        weight_decay=0.01,
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
        
        

    for epoch in tqdm(range(1, epoch_num+1)):
        loss, accuracy = train(
            train_graphs,
            memory,
            gnn,
            link_pred,
            neighbor_loader,
            optimizer,
            criterion,
            assoc,
            BATCH,
            device
            )
        logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        model = [memory, gnn, link_pred, neighbor_loader]
        torch.save(model, models_dir + f"new_models+{epoch}.pt")
            
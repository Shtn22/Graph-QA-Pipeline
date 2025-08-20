import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.utils.graph_retrieval import retrive_on_graphs
from torch_geometric.data.data import Data
import warnings

warnings.filterwarnings("ignore")

path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        try:
            with open(f'{cached_desc}/{index}.txt', 'r', encoding='utf-8') as f:
                desc = f.read()
        except FileNotFoundError:
            desc = ""
        label = ('|').join(data['answer']).lower()
        return {
            'id': index, 'question': question, 'label': label,
            'graph': graph, 'desc': desc,
        }

    def get_idx_split(self):
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

def preprocess(topk_nodes=25, num_hops=3, topk_subgraphs=20, topk_edges=60):
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    q_embs = torch.load(f'{path}/q_embs.pt')
    print("Starting final graph retrieval and caching process...")
    for index in tqdm(range(len(dataset)), desc="Processing graphs"):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        q_emb = q_embs[index]
        subg, desc = retrive_on_graphs(
            graph, q_emb, nodes, edges, 
            topk_nodes=topk_nodes, num_hops=num_hops, 
            topk_subgraphs=topk_subgraphs, topk_edges=topk_edges
        )
        if subg.num_nodes > 0:
             data = Data(x=subg.x, edge_index=subg.edge_index, edge_attr=subg.edge_attr, 
                        question_node=q_emb.repeat(subg.x.size(0), 1),
                        question_edge=q_emb.repeat(subg.edge_attr.size(0), 1) if subg.edge_attr is not None and subg.edge_attr.numel() > 0 else None,
                        num_nodes=subg.num_nodes)
        else:
            embedding_dim = q_emb.shape[0]
            data = Data(x=torch.empty(0, embedding_dim), 
                        edge_index=torch.empty((2,0), dtype=torch.long),
                        edge_attr=torch.empty(0, embedding_dim),
                        question_node=torch.empty(0, embedding_dim),
                        question_edge=torch.empty(0, embedding_dim),
                        num_nodes=0)
            
        torch.save(data, f'{cached_graph}/{index}.pt')
        with open(f'{cached_desc}/{index}.txt', 'w', encoding='utf-8') as f:
            f.write(desc)

if __name__ == '__main__':
    preprocess(topk_nodes=25, num_hops=3, topk_subgraphs=20, topk_edges=60)
    dataset = WebQSPDataset()
    print(f"Successfully preprocessed and cached all graphs. Ready for training.")
    print(f"Example of first item: {dataset[0]}")
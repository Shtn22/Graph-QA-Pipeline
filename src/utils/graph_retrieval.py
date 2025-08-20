import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from tqdm import tqdm

from src.utils.lm_modeling import load_sbert, sber_text2embedding

def relabel_nodes(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    used_nodes = torch.unique(edge_index)
    assoc = torch.full((num_nodes,), -1, dtype=torch.long)
    assoc[used_nodes] = torch.arange(used_nodes.size(0))
    new_edge_index = assoc[edge_index]
    return used_nodes, new_edge_index

print("Initializing model for graph retrieval utility...")
model, tokenizer, device = load_sbert()

def get_trunk_triplets(subgraph_nodes, subgraph_edge_indices, textual_nodes, textual_edges):
    if subgraph_edge_indices.size(1) == 0:
        node_texts = textual_nodes.loc[subgraph_nodes.tolist()]["node_attr"].tolist()
        return ", ".join(node_texts)
    node_texts_map = textual_nodes["node_attr"].to_dict()
    edge_attr_map = {tuple(row[['src', 'dst']]): row['edge_attr'] for _, row in textual_edges.iterrows()}
    flatten_graph_components = []
    for head, tail in subgraph_edge_indices.t().tolist():
        edge_text = edge_attr_map.get((head, tail), "")
        head_text, tail_text = node_texts_map.get(head, ""), node_texts_map.get(tail, "")
        flatten_graph_components.append(f"{head_text}, {edge_text}, {tail_text}")
    return ". ".join(flatten_graph_components)

def merge_graphs(graph_list):
    if not graph_list: return Data()
    if len(graph_list) == 1: return graph_list[0]
    x, edge_index, edge_attr, node_offset = [], [], [], 0
    for graph in graph_list:
        x.append(graph.x)
        if graph.edge_index is not None and graph.edge_index.numel() > 0:
            edge_index.append(graph.edge_index + node_offset)
        if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
            edge_attr.append(graph.edge_attr)
        node_offset += graph.num_nodes
    final_x = torch.cat(x, dim=0)
    final_edge_index = torch.cat(edge_index, dim=1) if edge_index else torch.empty((2, 0), dtype=torch.long)
    final_edge_attr = torch.cat(edge_attr, dim=0) if edge_attr else torch.empty((0, final_x.size(1)))
    return Data(x=final_x, edge_index=final_edge_index, edge_attr=final_edge_attr)

def retrive_on_graphs(graph, q_emb, textual_nodes, textual_edges, topk_nodes=25, num_hops=3, topk_subgraphs=20, topk_edges=60):
    if graph.num_nodes == 0:
        return Data(x=torch.empty(0, q_emb.shape[0]), edge_index=torch.empty((2,0), dtype=torch.long)), ""
        
    node_id_to_text_map = dict(zip(textual_nodes['node_id'], textual_nodes['node_attr']))
    relation_map = {tuple(row[['src', 'dst']]): row['edge_attr'] for _, row in textual_edges.iterrows()}
    all_edges_list = []
    for src, dst in graph.edge_index.t().tolist():
        relation = relation_map.get((src, dst)) or relation_map.get((dst, src), "related to")
        all_edges_list.append({'src': src, 'dst': dst, 'edge_attr': relation})
    aligned_textual_edges = pd.DataFrame(all_edges_list)

    node_sims = torch.nn.CosineSimilarity(dim=-1)(q_emb.cpu(), graph.x)
    num_nodes_to_select = min(topk_nodes, graph.num_nodes)
    if num_nodes_to_select == 0:
        return Data(x=torch.empty(0, q_emb.shape[0]), edge_index=torch.empty((2,0), dtype=torch.long)), ""
    top_node_indices = torch.topk(node_sims, k=num_nodes_to_select, largest=True).indices
    subgraph_data = []
    for node_id in top_node_indices:
        sub_nodes, sub_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=node_id.item(), num_hops=num_hops, edge_index=graph.edge_index,
            relabel_nodes=False, num_nodes=graph.num_nodes
        )
        sub_textual_edges = aligned_textual_edges[edge_mask.numpy()]
        flattened_text = get_trunk_triplets(sub_nodes, sub_edge_index, textual_nodes, sub_textual_edges)
        subgraph_data.append({"center_node_id": node_id.item(), "text": flattened_text})

    subgraph_texts = [s['text'] for s in subgraph_data]
    subgraph_text_embs = sber_text2embedding(model, tokenizer, device, subgraph_texts)
    subgraph_sims = torch.nn.CosineSimilarity(dim=-1)(q_emb.cpu(), subgraph_text_embs)
    num_subgraphs_to_select = min(topk_subgraphs, len(subgraph_sims))
    top_subgraph_indices = torch.topk(subgraph_sims, k=num_subgraphs_to_select, largest=True).indices
    
    graphs_to_merge, original_indices_for_merge, textual_edges_for_merge = [], [], []
    for idx in top_subgraph_indices:
        center_node = subgraph_data[idx]['center_node_id']
        sub_nodes, sub_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=center_node, num_hops=num_hops, edge_index=graph.edge_index,
            relabel_nodes=True, num_nodes=graph.num_nodes
        )
        sub_x, sub_edge_attr = graph.x[sub_nodes], graph.edge_attr[edge_mask]
        graphs_to_merge.append(Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr))
        original_indices_for_merge.append(sub_nodes)
        textual_edges_for_merge.append(aligned_textual_edges[edge_mask.numpy()])

    merged_graph = merge_graphs(graphs_to_merge)
    if textual_edges_for_merge:
        merged_textual_edges = pd.concat(textual_edges_for_merge, ignore_index=True)
    if original_indices_for_merge:
        merged_graph_node_map = torch.cat(original_indices_for_merge)

    if merged_graph.num_nodes == 0 or merged_graph.num_edges == 0:
       return Data(x=torch.empty(0, q_emb.shape[0]), edge_index=torch.empty((2,0), dtype=torch.long)), ""

    unique_indices = merged_textual_edges.drop_duplicates(subset=['src', 'dst', 'edge_attr']).index
    
    unique_edge_attr = merged_graph.edge_attr[unique_indices]
    unique_textual_edges = merged_textual_edges.iloc[unique_indices]
       
    edge_sims = torch.nn.CosineSimilarity(dim=-1)(q_emb.cpu(), unique_edge_attr)
    num_edges_to_select = min(topk_edges, len(unique_textual_edges))
    top_unique_indices = torch.topk(edge_sims, k=num_edges_to_select, largest=True).indices
    
    final_textual_edges_df = unique_textual_edges.iloc[top_unique_indices.tolist()]

    original_top_indices = unique_indices[top_unique_indices]
    pruned_edge_index = merged_graph.edge_index[:, original_top_indices]
    pruned_edge_attr = merged_graph.edge_attr[original_top_indices]
    
    final_nodes_in_merged_space, final_edge_index = relabel_nodes(pruned_edge_index, num_nodes=merged_graph.num_nodes)
    final_x = merged_graph.x[final_nodes_in_merged_space]
    final_pruned_graph = Data(x=final_x, edge_index=final_edge_index, edge_attr=pruned_edge_attr, num_nodes=final_x.size(0))

    final_textual_edges_df['src_text'] = final_textual_edges_df['src'].map(node_id_to_text_map)
    final_textual_edges_df['dst_text'] = final_textual_edges_df['dst'].map(node_id_to_text_map)
    src_nodes = final_textual_edges_df[['src', 'src_text']].rename(columns={'src': 'node_id', 'src_text': 'node_attr'})
    dst_nodes = final_textual_edges_df[['dst', 'dst_text']].rename(columns={'dst': 'node_id', 'dst_text': 'node_attr'})
    final_textual_nodes_df = pd.concat([src_nodes, dst_nodes]).drop_duplicates(subset=['node_id']).reset_index(drop=True)
    
    desc_nodes = final_textual_nodes_df.to_csv(index=False, columns=['node_id', 'node_attr'])
    desc_edges = final_textual_edges_df.to_csv(index=False, columns=['src_text', 'edge_attr', 'dst_text'])
    desc = desc_nodes + '\n' + desc_edges

    return final_pruned_graph, desc
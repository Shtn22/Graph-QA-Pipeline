import os
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.lm_modeling import load_sbert, sber_text2embedding

PATH = 'dataset/webqsp'

def pre_encode_all_entities():
   
    print("Starting one-time pre-encoding process for WebQSP...")
    os.makedirs(PATH, exist_ok=True)

    print("Loading rmanluo/RoG-webqsp dataset...")
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    print(f"Dataset loaded with {len(dataset)} total entries.")

    print("Loading sentence transformer model: sentence-transformers/all-roberta-large-v1")
    model, tokenizer, device = load_sbert()

    print("Aggregating unique node texts...")
    unique_node_texts = set()
    for i in tqdm(range(len(dataset)), desc="Aggregating Node Texts"):
        for tri in dataset[i]['graph']:
            h, _, t = tri
            unique_node_texts.add(h.lower())
            unique_node_texts.add(t.lower())
            
    unique_node_list = list(unique_node_texts)
    print(f"Found {len(unique_node_list)} unique node texts.")

    print("Encoding unique node texts...")
    node_embeddings = sber_text2embedding(model, tokenizer, device, unique_node_list)

    print("Saving node embeddings and mapping to disk...")
    node_text_to_idx = {text: i for i, text in enumerate(unique_node_list)}
    torch.save(node_embeddings, f'{PATH}/unique_node_embeddings.pt')
    with open(f'{PATH}/node_text_to_idx.json', 'w') as f:
        json.dump(node_text_to_idx, f)
    print("Node assets saved successfully.")
    
    del node_embeddings
    del unique_node_list
    del node_text_to_idx

    print("\nAggregating unique contextual edge texts...")
    unique_contextual_edge_texts = set()
    for i in tqdm(range(len(dataset)), desc="Aggregating Edge Texts"):
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h, t = h.lower(), t.lower()
            unique_contextual_edge_texts.add(f"{h} {r} {t}")
            unique_contextual_edge_texts.add(f"{t} {r} {h}")
            
    unique_edge_list = list(unique_contextual_edge_texts)
    print(f"Found {len(unique_edge_list)} unique contextual edge texts.")

    print("Encoding unique contextual edge texts in chunks to conserve RAM...")
    
    chunk_size = 1000000  
    all_edge_embeddings = []

    for i in tqdm(range(0, len(unique_edge_list), chunk_size), desc="Processing Edge Chunks"):
        chunk = unique_edge_list[i:i + chunk_size]
        chunk_embeddings = sber_text2embedding(model, tokenizer, device, chunk)
        all_edge_embeddings.append(chunk_embeddings)

    print("Concatenating edge embedding chunks...")
    final_edge_embeddings = torch.cat(all_edge_embeddings, dim=0)

    print("Saving edge embeddings and mapping to disk...")
    edge_text_to_idx = {text: i for i, text in enumerate(unique_edge_list)}
    torch.save(final_edge_embeddings, f'{PATH}/unique_contextual_edge_embeddings.pt')
    with open(f'{PATH}/contextual_edge_text_to_idx.json', 'w') as f:
        json.dump(edge_text_to_idx, f)
    
    print("Edge assets saved successfully.")
    print("\nPre-encoding process completed successfully.")


if __name__ == '__main__':
    pre_encode_all_entities()
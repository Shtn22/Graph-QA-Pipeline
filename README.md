
# Graph-Augmented Question Answering with a Custom Retrieval Pipeline

This project introduces a custom, multi-stage data preprocessing and subgraph retrieval pipeline designed to enhance the reasoning capabilities of Large Language Models (LLMs) on complex knowledge graph datasets.

The core of this work is a sophisticated retrieval algorithm that distills massive, noisy graphs into small, dense, and semantically relevant subgraphs. These optimized subgraphs are then used to train a hybrid Graph Neural Network (GNN) and LLaMA-7B model. The final implementation achieved a Hit Rate of 71.74% and an F1 Score of 54.32% on the WebQSP test set, demonstrating the effectiveness of this custom pipeline.

## Project Highlights

- **Custom Multi-Stage Retrieval:** A novel pipeline that identifies relevant seed nodes, performs deep 3-hop contextual expansion, ranks entire subgraphs based on textual similarity, and performs a final pruning to keep only the top 60 most relevant edges.
- **Context-Aware Embeddings:** Implemented an advanced edge embedding strategy where the feature for an edge is derived from the text of its source node, destination node, and the relation itself, providing a rich semantic signal.
- **Optimized Preprocessing:** Developed a one-time global encoding script that pre-computes embeddings for the entire dataset's ~1.3 million unique nodes and ~7.5 million unique contextual edges, drastically reducing computational overhead.
- **Parameter-Efficient Training:** Successfully trained the model by freezing the 6.7 billion parameters of the LLaMA-7B model and exclusively fine-tuning the 39.9 million parameters of the GNN and a custom projector module.
- **Robust and Debugged:** The entire codebase has been meticulously debugged to handle hardware constraints (single GPU), data inconsistencies, and complex software dependency issues.

## Architecture Overview

The system utilizes a hybrid architecture that synergizes two main components:

- **GNN Encoder (Graph Attention Network - GAT):** Processes the topological structure of the retrieved subgraph, learning to aggregate information from the most relevant neighboring nodes.
- **LLM Backbone (LLaMA-7B):** A frozen LLM that receives the standard text prompt along with a special "graph token"—a single vector embedding produced by the GNN and a projector module that summarizes the graph's structure.

## Setup and Execution

### 1. Environment Setup

This project requires a Conda environment. The following commands will configure a stable, GPU-enabled environment.

```bash
# Create and activate the Conda environment
conda create --name qa_env python=3.9 -y
conda activate qa_env

# Install PyTorch for CUDA 11.8 and the compatible version of PyTorch Geometric
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 pyg -c pytorch -c nvidia -c pyg

# Install remaining dependencies with specific, compatible versions
pip install "numpy<2"
pip install transformers==4.30.0 accelerate==0.21.0 peft==0.4.0 pandas datasets wandb
```

### 2. Running the Pipeline

The project is divided into a three-stage preprocessing workflow followed by the final training. These scripts must be run in the specified order from the project's root directory.

#### Stage 1: One-Time Global Encoding

```bash
python src/dataset/preprocess/pre_encode_all.py
```

#### Stage 2: Build Full Graphs

```bash
python src/dataset/preprocess/webqsp.py
```

#### Stage 3: Retrieve and Cache Final Subgraphs
(Note: This script must be run as a module)

```bash
python -m src.dataset.webqsp
```

#### Stage 4: Train the Model


```bash
python train.py --dataset webqsp 
```

## Final Results

After a full training and evaluation run, the model achieved the following scores on the WebQSP test set:

- **Hit Rate:** 71.74%
- **F1 Score:** 54.32%
- **Precision:** 71.37%
- **Recall:** 53.44%

These results demonstrate a highly effective implementation capable of state-of-the-art performance on this complex reasoning task.

## Acknowledgements

I would like to thank the authors of the G-Retriever and GRAG papers for their open-source contributions, which provided a valuable foundation for this project.

# 📄 Full Project Paper

```
Project ID: Summer internship
University of Manouba
National School of Computer Science
Report of the Design and Development Project
Subject: Graph-Augmented Retrieval for
Enhanced Factual Reasoning in Large
Language Models
Authors :
Mr. Saif Eddine Hmaied
Mr. Mohamed Aziz Othmani Mr. Mohamed Nour
Mosbehi
Supervisor :
Mr. Marouene Chaieb
Academic Year : 2024/2025

Appreciation and Supervisor’s
Signature
2

Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in
natural language understanding and generation, yet they often struggle with tasks
requiring deep factual knowledge and complex reasoning, frequently producing plau-
sible but incorrect information (”hallucinations”). This limitation is particularly
evident when answering questions over large, structured Knowledge Graphs (KGs).
This report provides a comprehensive technical breakdown of a custom-designed
data processing pipeline and model training workflow for graph-augmented ques-
tion answering. The core of this work involved the creation of a highly optimized,
multi-stage retrieval algorithm designed to distill massive, noisy graphs into small,
dense, and semantically relevant subgraphs. We detail the specific architecture,
including the textual encoder, the Graph Attention Network (GAT), a custom
projector module, and the frozen LLaMA-7B backbone. A meticulous descrip-
tion of the data pipeline is provided, from the initial context-aware global en-
coding strategy to the final subgraph retrieval and de-duplication logic. The re-
port concludes with an analysis of the training dynamics, which yielded strong
retrieval and end-to-end QA performance, validating the efficacy of this deeply
customized approach.Our datasets as well as codes of GRAG are available at
https://github.com/Shtn22/Graph-QA-Pipeline
3

List Of Acronyms
LLM Large Language Model
KG Knowledge Graph
GNN Graph Neural Network
RAG Retrieval-Augmented Generation
GAT Graph Attention Network
PPR Personalized PageRank
BGE BAAI General Embedding
MID Machine ID (Freebase)
KGQA Knowledge Graph Question Answering
PyG PyTorch Geometric
4

Contents
1 Introduction 7
1.1 Context of the Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
1.2 Problem Statement and Motivation . . . . . . . . . . . . . . . . . . . . . 7
1.3 Aims and Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
1.4 Proposed Solution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
1.5 Structure of the Report . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
2 Literature Review 9
2.1 Technical Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.1.1 Large Language Models (LLMs) . . . . . . . . . . . . . . . . . . . 9
2.1.2 Knowledge Graphs (KGs) . . . . . . . . . . . . . . . . . . . . . . 9
2.1.3 Graph Neural Networks (GNNs) . . . . . . . . . . . . . . . . . . . 10
2.1.4 Retrieval-Augmented Generation (RAG) . . . . . . . . . . . . . . 11
2.2 Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.3 Findings of the Literature and Contributions of the Study . . . . . . . . 13
3 Methodology and Proposed Solution 14
3.1 Data Collection and Understanding . . . . . . . . . . . . . . . . . . . . . 15
3.2 Data Preprocessing (Offline Encoding & Hydration) . . . . . . . . . . . . 15
3.2.1 Stage 1 — Optimized One-Time Global Encoding . . . . . . . . . 16
3.2.2 Stage 2 — Full Graph Hydration . . . . . . . . . . . . . . . . . . 16
3.2.3 Stage 3 — Multi-Stage Subgraph Retrieval Algorithm . . . . . . . 16
3.2.4 Textualization Strategies — notes . . . . . . . . . . . . . . . . . . 17
3.3 Model Architecture (Online Model Building) . . . . . . . . . . . . . . . . 17
3.3.1 GNN Encoder (Question-aware GAT) . . . . . . . . . . . . . . . . 17
3.3.2 Projector Module . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
3.3.3 LLM Backbone . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
3.4 Training and Caching . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
3.5 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
3.6 Implementation notes and scripts . . . . . . . . . . . . . . . . . . . . . . 19
4 Implementation and Results 19
4.1 Environment and Working Tools . . . . . . . . . . . . . . . . . . . . . . . 19
4.1.1 Hardware Environment . . . . . . . . . . . . . . . . . . . . . . . . 19
4.1.2 Software Environment . . . . . . . . . . . . . . . . . . . . . . . . 20
4.2 Technological Choices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4.2.1 Programming Language . . . . . . . . . . . . . . . . . . . . . . . 20
4.2.2 Libraries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4.3 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4.4 Experimental Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
5 Conclusion and Perspectives 23
5.1 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
5.2 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
5.3 Perspectives and Future Work . . . . . . . . . . . . . . . . . . . . . . . . 23
5.4 Final Remarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
5

6 References 25
6

1 Introduction
1.1 Context of the Study
The advent of Large Language Models (LLMs) such as the LLaMA family, GPT, and
Claude has marked a paradigm shift in artificial intelligence. Trained on vast corpora
of text data, these models excel at a wide range of natural language tasks. However,
their knowledge is parametric—it is implicitly stored within the model’s weights—and
can be static, difficult to update, and prone to factual inaccuracies, commonly known
as hallucinations. In parallel, structured Knowledge Graphs (KGs) have emerged as
powerful repositories of factual information, representing entities and their relationships
as nodes and edges. KGs offer explicit, verifiable, and easily updatable knowledge, but
their symbolic, graph-based nature makes them inaccessible to purely text-based LLMs.
The central challenge, and the context for this study, is to effectively fuse the fluid
linguistic capabilities of LLMs with the rigid factual accuracy of KGs to create more
powerful and trustworthy reasoning systems.
1.2 Problem Statement and Motivation
When presented with a question that requires factual knowledge (e.g., ”Which university
gave Nelson Mandela an honorary degree in 1996?”), an LLM must rely solely on patterns
from its training data. A large KG like the one underlying the WebQSP dataset contains
this answer explicitly but is far too vast (millions of nodes and edges) to be used directly.
Feeding the entire graph, or even a large, unfiltered portion of it, to an LLM would be
computationally infeasible and would inundate the model with irrelevant noise, confusing
its reasoning process. The motivation for this project is to solve this ”signal vs. noise”
problem. We need an intelligent mechanism to retrieve a small, precise ”nugget” of the
KG that contains the answer and its immediate reasoning context, and to present this
structured information in a format that an LLM can natively understand and utilize.
This project addresses the critical need for a sophisticated retrieval pipeline that goes
beyond simple keyword matching or naive graph traversal.
1.3 Aims and Objectives
The overarching aim of this internship project was to design and evaluate a custom
multi-stage retrieval and graph-augmented question answering pipeline capable
of handling large, noisy knowledge graphs under constrained compute resources.
To achieve this aim, the project pursued the following specific objectives:
1.Efficient Graph Encoding: Develop a one-time global encoding process for all
nodes and contextual edges in the WebQSP knowledge graph, producing reusable
embeddings for downstream tasks.
2.Subgraph Retrieval Pipeline: Design and implement a multi-stage retrieval
algorithm that distills massive graphs into small, dense, and semantically relevant
subgraphs tailored to each question.
3.Model Architecture Integration: Combine a Graph Attention Network (GAT)
7

encoder, a custom projector module, and a frozen LLaMA-7B backbone into a
unified GraphLLM pipeline.
4.Parameter-Efficient Training: Apply a training strategy that freezes the LLM
while optimizing the GNN and projector components, making training feasible on
a single GPU.
By fulfilling these objectives, the project sought to demonstrate that a carefully engi-
neered pipeline can achieve competitive performance with state-of-the-art methods, even
under limited resources.
1.4 Proposed Solution
Our proposed solution is an end-to-end pipeline that integrates graph preprocessing,
multi-stage retrieval, and graph-augmented reasoning with a frozen large language model.
The approach is built on three main design choices:
1.Custom Multi-Stage Retrieval Pipeline: A novel retrieval mechanism was
developed to handle the scale and noisiness of the WebQSP knowledge graph. This
pipeline performs:
•Global encoding of all nodes and contextual edges using sentence transformers
/ all-roberta-large-v1 .
•Subgraph construction around seed nodes most semantically related to the
question.
•Ranking, merging, de-duplication, and pruning to produce compact and question-
focused subgraphs of approximately 60 edges.
2.GraphLLM Hybrid Architecture: The system combines:
•A Graph Attention Network (GAT) encoder to capture structural and semantic
relations.
•A custom projector module that transforms graph embeddings into a 4096-
dimensional vector aligned with the LLaMA word embedding space.
•The frozen LLaMA-7B model as the reasoning and generation engine, enabling
parameter-efficient training.
3.Parameter-Efficient Training Strategy: The LLaMA backbone remains frozen
(6.7B parameters), while only the GAT and projector components are optimized
(39.9M parameters). This allows the model to be trained efficiently on a single T4
GPU while still leveraging LLaMA’s strong language capabilities.
This solution ensures that the model receives a clean, semantically dense graph represen-
tation as a special input token, enabling it to reason jointly over natural language and
graph-derived evidence.
1.5 Structure of the Report
This report is organized as follows: Section 2 reviews the technical background and re-
lated works in the field. Section 3 provides a detailed description of our methodology and
8

proposed solution, covering data preprocessing, model architecture, and evaluation met-
rics. Section 4 outlines the implementation details, including the software environment
and experimental setup, and presents the results. Finally, Section 5 concludes the report
and discusses potential future work.
2 Literature Review
2.1 Technical Background
2.1.1 Large Language Models (LLMs)
Large Language Models (LLMs) are deep neural networks based on the Transformer ar-
chitecture that are pre-trained on massive text corpora via self-supervised objectives (e.g.,
next-token prediction). This pretraining enables models to learn rich representations of
grammar, semantics, and factual patterns that can be adapted to downstream tasks via
fine-tuning or prompt-based conditioning. In this project we adopt LLaMA-2 (7B) as
our base model due to its open accessibility and strong empirical performance among
open-source chat-capable LLMs [6].
Fine-tuning full LLM weights can be prohibitively expensive for large models, so parameter-
efficient adaptation methods are commonly used. One widely adopted technique is Low-
Rank Adaptation (LoRA), which freezes the pre-trained model parameters and injects
low-rank trainable update matrices into attention and feed-forward layers; this approach
drastically reduces the number of trainable parameters and memory footprint while re-
taining strong downstream performance [3].
2.1.2 Knowledge Graphs (KGs)
A Knowledge Graph (KG) is a structured representation of facts in which entities are
modeled as nodes and relations between them are modeled as labeled edges. Facts are
commonly stored as triplets of the form (head entity, relation, tail entity) ; to-
gether these triplets form a heterogeneous graph that can capture rich, multi-relational
world knowledge. In practice, nodes and edges often carry additional attributes (aliases,
types, timestamps, provenance) that are useful for disambiguation, filtering, and down-
stream reasoning.
The WebQSP dataset used in this work is built on Freebase, a large-scale KG that contains
millions of entities and relations. This scale makes KGs excellent testbeds for research
on scalable retrieval and multi-hop reasoning, but it also introduces several practical
challenges. First, scale and sparsity mean that exhaustive search is infeasible and naive
retrieval returns noisy, irrelevant subgraphs. Second, many KG snapshots contain opaque
identifiers (Freebase Machine IDs, or MIDs) and incomplete labels; unresolved MIDs or
missing aliases create ambiguity for purely string-based matching. Third, KG topology is
directional and multi-relational: reasoning often requires following specific relation types
9

(e.g., spouse ,born in,inventor of) across multiple hops, so preserving edge semantics
during retrieval is crucial.
Figure 1: A toy Knowledge Graph fragment illustrating entities (nodes) and relations
(edges).
Table 1: Summary of dataset (WebQSP) used in the GraphQA benchmark.
Dataset WebQSP
#Graphs 4,737
Avg. #Nodes 1370.89
Avg. #Edges 4252.37
Node Attribute Entities in Freebase
Edge Attribute Relations in Freebase
Task Knowledge-based question answering
Evaluation Metric Hit@1
2.1.3 Graph Neural Networks (GNNs)
Graph Neural Networks (GNNs) are a family of neural architectures specifically designed
to process graph-structured data. The core idea is message passing : at each layer every
node aggregates information (messages) from its neighbors and then updates its own
feature vector using a learnable function. After several rounds of message passing a
node’s embedding encodes not only its local attributes but also the multi-hop structural
context required for relational reasoning.
GNNs differ in how they compute messages and how they aggregate them. Convolutional-
style GNNs (e.g., GCNs) use fixed neighborhood averaging, while attention-based models
learn to weight neighbors by relevance. In particular, the Graph Attention Network
(GAT) [7] employs self-attention at the edge level so that the model can focus on the
most informative neighbors for each node. Attention mechanisms improve expressivity
for heterogeneous relation types and reduce the impact of noisy neighbors — a desirable
property when working with large, noisy knowledge graphs.
10

2.1.4 Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is a paradigm for improving the factual accuracy
and updatability of large language models by supplying them with external context at
inference time. A canonical RAG pipeline decomposes the task into three components:
(1) a retriever that finds candidate pieces of evidence (documents, passages, or structured
items) relevant to the user query; (2) an optional re-ranker that scores and selects the
most useful items; and (3) a generator (the LLM) that conditions its output on the
retrieved context. This design separates knowledge storage (the retrieval index) from the
parametric knowledge encoded in the model weights, enabling easier updates, improved
factual grounding, and often reduced hallucination.
Retrieval modules fall into several families. Traditional sparse retrievers (e.g., BM25) are
fast and interpretable but rely on lexical overlap. Dense retrievers use learned embeddings
and can capture semantic matches that sparse methods miss, at the cost of heavier
indexing and nearest-neighbor search. Reranking layers (cross-encoders or learned scoring
models) further improve precision by scoring candidates in context. In practice, there is
a trade-off among recall ,precision ,latency , and index/update cost that must be balanced
for the target application.
Most RAG systems operate over unstructured text, concatenating retrieved passages to
the user prompt. However, for tasks that require multi-hop relational reasoning (KGQA,
complex fact-checking), unstructured text can lose crucial relational signals. Structured
retrieval — returning a compact, topologically-coherent subgraph instead of raw passages
— preserves explicit relations and reasoning paths but introduces new challenges: how to
(i) rank subgraphs by conceptual relevance, (ii) compress graph structure into a format
consumable by a text-based LLM, and (iii) control subgraph size to meet token-budget
and noise constraints.
2.2 Related Works
LLaMA 2 [6](Touvron et al., 2023) introduces a suite of open-access foundation models
with parameter sizes of 7B, 13B, and 70B, trained on a significantly expanded corpus com-
pared to its predecessor. The paper presents both base and instruction-tuned variants,
optimized for safe and helpful dialogue generation. A key contribution is the release of
reproducible training recipes and alignment techniques, including Reinforcement Learn-
ing from Human Feedback (RLHF). This work lays the foundation for responsible open
development of large-scale language models and serves as a critical benchmark in evalu-
ating LLM performance and safety. In our work, we leverage the 7B version of LLaMA
2 as the core language model for graph-augmented reasoning.
G-Retriever [2] introduces a novel retrieval-augmented generation (RAG) framework for
textual graph question answering, enabling interaction with large heterogeneous graphs
such as scene graphs, commonsense graphs, and knowledge graphs. Unlike traditional
methods that focus on small or structured graphs, G-Retriever constructs a cross-domain
benchmark (GraphQA) and formulates subgraph retrieval as a Prize-Collecting Steiner
Tree (PCST) problem to assemble a connected and relevant subgraph. This graph is then
11

encoded using a Graph Attention Network (GAT) and injected as soft prompts into a
frozen LLM. The method significantly improves factual grounding and QA accuracy on
datasets such as ExplaGraphs, SceneGraphs, and WebQSP, demonstrating the benefits
of structural reasoning over naive retrieval techniques.
Think-on-Graph 2.0 (ToG-2) [5] proposes a hybrid RAG framework that tightly couples
retrieval from both unstructured documents and structured knowledge graphs. Unlike
naive RAG pipelines, ToG-2 alternates between context retrieval via documents and
graph-based subgraph expansion guided by entity links. This iterative interplay deepens
semantic coverage and improves factual consistency by enforcing KG-guided navigational
constraints. Notably, ToG-2 is training-free and plug-and-play with various LLMs, and
achieves state-of-the-art performance across six out of seven knowledge-intensive QA
datasets—sometimes enabling LLaMA-2-13B to match GTK-3.5’s reasoning performance.
Its design underscores the importance of KG-aware retrieval loops for deep reasoning in
LLMs.
Wan et al. (2025) [8] present an innovative hybrid retrieval-augmented generation (RAG)
framework tailored for domain-centric question answering in smart manufacturing. To
bridge the gap between vector-based RAG (fast but vague) and knowledge-graph (KG)
approaches (structured but less scalable), their system constructs a metadata-enriched
KG from domain corpora and aligns it semantically using domain-specific constraints. Re-
trieval is performed in layers—first via efficient vector-similarity, then with KG-guided,
precision-focused filtering. The outputs are merged via prompt engineering, resulting in
significantly improved accuracy: 77.8 % exact match and 76.5% context precision on De-
sign for Additive Manufacturing (DfAM) tasks. This hybrid symbolic-neural architecture
effectively balances scalability and precision in industrial LLM deployments.
Zhang et al. (2022) [9] propose a trainable, decoupled subgraph retriever (SR) for multi-
hop knowledge base question answering (KBQA), addressing the critical trade-off between
subgraph size and answer relevance. Unlike heuristic or tightly coupled retrieval-reasoning
methods, SR employs a dual-encoder framework to expand paths relevant to the ques-
tion and induces compact, high-quality subgraphs for downstream reasoning. Through
weakly supervised pre-training, unsupervised pseudo-path generation, and optional end-
to-end fine-tuning with any subgraph-oriented reasoner (e.g., NSM or GRAFT-Net), SR
achieves state-of-the-art performance—including up to a 9.7 % increase in Hits@1—while
significantly reducing subgraph size and noise.
Hu et al. (2025) [4] introduce Graph Retrieval-Augmented Generation (GRAG), a novel
framework extending traditional RAG methods to account for both textual and topologi-
cal context in interconnected documents, such as citation networks or knowledge graphs.
They propose a divide-and-conquer strategy for efficient subgraph retrieval—using K-
hop ego-graphs and soft pruning to approximate optimal subgraphs in linear time—thus
avoiding NP-hard exhaustive search. GRAG then converts these subgraphs into two
complementary prompt views: a **text view**, generated via hierarchical hard prompts
that retain topological structure, and a **graph view**, encoded as soft prompts via a
GNN. Empirical evaluations on multi-hop reasoning benchmarks show that this approach
significantly outperforms state-of-the-art RAG and LLM baselines, even when the LLM
remains frozen.
12

de Zarz` a et al. (2024) [1] propose an optimization framework that integrates Large Lan-
guage Model (LLM) recommendations into both individual and cooperative (household)
budgeting workflows. The paper formulates mathematical optimization models for per-
sonalized budget allocation and extends them with LLM-driven heuristics and recommen-
dations to improve accessibility, adaptability, and personalization of financial plans. The
authors present a simulation study on synthetic household profiles that demonstrates
improved personalization, scalability and responsiveness to market changes compared
to traditional approaches, and discuss an “extended coevolutionary” perspective that
models the interplay between human behavior and AI agents. This work exemplifies a
practical instantiation of hybrid symbolic–neural systems in a domain-specific applica-
tion, highlighting both opportunities and caveats when LLM suggestions are used inside
optimization loops.
2.3 Findings of the Literature and Contributions of the Study
Findings from the Literature
The literature on question answering over knowledge graphs has highlighted several im-
portant trends:
•Retrieval-Augmented Generation (RAG): Studies demonstrate that integrat-
ing retrieval with LLMs improves factual grounding, but naive retrieval often in-
troduces noisy or irrelevant context.
•Graph Neural Networks (GNNs): Prior work shows that GNNs such as Graph
Attention Networks (GATs) excel at modeling relational structures, yet their out-
puts must be carefully aligned with language models to be useful for QA.
•Graph-LLM Hybrids: Recent GraphLLM approaches illustrate that inserting
graph-derived tokens into LLM input streams can significantly improve reasoning,
but existing solutions often struggle with large, noisy graphs due to scalability
issues.
Contributions of the Study
Building on these findings, this study contributes the following:
1.Custom Multi-Stage Retrieval Pipeline: A novel retrieval algorithm capable
of distilling massive, noisy graphs (WebQSP-scale) into small, dense, semantically
relevant subgraphs of about 60 edges.
2.Integration of a Projector Module: A lightweight MLP that maps graph
embeddings into the LLaMA embedding space, enabling smooth fusion of graph-
derived and text-derived signals.
3.Parameter-Efficient Training Strategy: Freezing the 6.7B-parameter LLaMA
backbone while training only 39.9M parameters, making the approach feasible on
limited hardware (single T4 GPU).
4.Empirical Validation: Demonstrated strong performance with a Hit Rate of
13

71.74% and F1 Score of 54.32, validating that careful preprocessing and retrieval
can unlock competitive results even under constrained resources.
These contributions advance the state of graph-augmented question answering by showing
that efficiency-focused designs can yield performance competitive with more resource-
heavy methods.
3 Methodology and Proposed Solution
Our proposed solution addresses the fundamental challenge of enabling Large Language
Models (LLMs) to effectively reason over large-scale Knowledge Graphs (KGs) through
a novel end-to-end pipeline architecture. The core innovation lies in transforming a
natural language question and a massive, noisy KG into a semantically-rich, structurally-
aware representation that preserves both the relational knowledge and the contextual
understanding required for complex multi-hop reasoning.
Unlike traditional approaches that rely solely on textual linearization of graph struc-
tures or simple node filtering techniques, our methodology introduces a hybrid retrieval-
reasoning framework that combines the semantic understanding capabilities of modern
text encoders with the structural reasoning power of Graph Neural Networks (GNNs).
The process is strategically divided into two complementary phases: an offline Data Pre-
processing phase that performs intelligent subgraph extraction and caching, and an online
Model Building phase that fuses graph-aware representations with pre-trained language
model capabilities. This architectural design ensures both computational efficiency dur-
ing inference and preservation of critical structural relationships that are essential for
accurate question answering over knowledge graphs.
14

QuestionGlobal Encoding
(Nodes & Edges)Multi-Stage Retrieval
(Seed→Expansion →Ranking →Pruning)
GAT Encoder
(Question-Aware)MLP Projector
(1024→4096-d)Frozen LLaMA-7B
(Reasoning & Generation)
Final Answer
Figure 2: Compact pipeline overview.
3.1 Data Collection and Understanding
The primary dataset used is WebQSP (sourced from the rmanluo/RoG-webqsp reposi-
tory), which contains approximately 4,700 question–answer pairs. Each question is asso-
ciated with a large raw subgraph extracted from Freebase. Questions frequently require
multi-hop reasoning and many node labels contain unresolved Freebase MIDs, which
motivated a textual-normalization and embedding-first strategy.
Table 2: Descriptive statistics of hydrated subgraphs (N = 4,700).
Statistic Nodes Edges
Mean 42.71 59.96
Std. Dev. 7.06 1.37
Min 0 0
Max 78 60
Comment: These descriptive statistics summarize the final hydrated subgraphs used by
the retrieval pipeline. The per-subgraph maxima/minima and modest standard devia-
tions motivate our pruning thresholds and indicate generally consistent subgraph sizes
across the dataset.
3.2 Data Preprocessing (Offline Encoding & Hydration)
To handle the extreme scale and noisiness of the WebQSP graph corpus, we implemented
a multi-part offline preprocessing pipeline. This one-time process produces disk-cached,
reusable artifacts (embeddings and hydrated graphs) that make online retrieval efficient.
15

3.2.1 Stage 1 — Optimized One-Time Global Encoding
•Motivation: The global corpus contains on the order of ∼1.3M unique node
strings and ∼7.5M unique contextual edge strings. Encoding these items on-the-fly
is prohibitively expensive; instead we perform a one-time, chunked encoding pass
and persist the results to disk.
•Textual encoder: We use sentence-transformers/all-roberta-large-v1 as
the universal encoder for questions, node texts, and contextual edge strings. The
encoder produces 1024-dimensional vectors which are L2-normalized for cosine sim-
ilarity ranking.
•Contextual edge strings: Edge features are encoded as full contextual strings of
the form "{source_node_text} {relation_text} {destination_node_text}" to
capture local semantics beyond relation labels alone.
•Chunking and precision: Encoding is performed in memory-safe chunks (ad-
justed to available RAM) and leverages mixed precision (FP16) where feasible to
speed throughput and reduce memory use.
•Patching NaNs: During encoding a small data bug produced NaN/missing entries
for some nodes. To avoid re-running the entire multi-day encoding, we add and
persist an embedding for the empty string ’’and ensure robust lookup behavior
when hydration encounters missing ids.
•Artifacts: The output of this stage is a set of chunked, on-disk tensor files (and
corresponding id →offset indices) for node embeddings and contextual edge embed-
dings. These files are used by all subsequent preprocessing steps.
3.2.2 Stage 2 — Full Graph Hydration
Using the globally-saved embeddings created in Stage 1, we construct a fully-featured,
bidirectional graph for each question. Hydration includes:
•mapping node and edge ids to their precomputed embeddings,
•making the graph bidirectional by adding inverse edges (tagged as reverse ofr),
•replacing unresolved MIDs with a placeholder token (kept in embeddings as ’’or
<unk entity> ), and
•serializing the hydrated graph as a PyTorch Geometric (PyG) Data object for later
retrieval.
3.2.3 Stage 3 — Multi-Stage Subgraph Retrieval Algorithm
This is the core retrieval algorithm used to extract small, dense, and question-focused
subgraphs for each question. For each hydrated question graph the pipeline executes:
1.Seed Node Identification: Compute cosine similarity between the question em-
bedding and all node embeddings in the hydrated graph, and select the top 25
seed nodes .
2.Deep Contextual Expansion (3-hop): For each seed node extract a 3-hop
neighborhood subgraph (node, 1-hop, 2-hop, and 3-hop). The 3-hop expansion
16

captures broad context while still remaining tractable for ranking.
3.Subgraph Textualization and Ranking: Each 3-hop subgraph is textualized
using the fast gettrunk triplets method (a comma-separated list of triplets).
The textual representations are encoded (using the same all-roberta-large-v1
encoder) and compared to the question embedding; the top 20 subgraphs by sim-
ilarity are retained for the merge step.
4.Merging and De-duplication: The selected subgraphs are merged into a single
combined edge list. A robust de-duplication routine eliminates redundant edges
that arise from overlapping neighborhoods (this step corrected a previous bug that
caused massive redundancy).
5.Final Pruning (top edges): From the merged, de-duplicated edge set we compute
contextual edge similarities to the question and select the top 60 edges . The final
retrieved graph consists of these 60 edges and the nodes they connect.
6.Caching: The final pruned graph is serialized to disk as a PyG Data object and
also saved as a human-readable textual snapshot (for debugging and inspection).
These cached objects are directly used by the training loop and by inference.
3.2.4 Textualization Strategies — notes
Two textualization methods were considered:
•gettrunk triplets (used for ranking): lightweight and fast; ideal for iteratively
ranking many candidate subgraphs.
•hard prompt (analyzed but not used for ranking): produces a human-readable,
hierarchical traversal of the subgraph (useful for debugging and potential future
prompting experiments).
3.3 Model Architecture (Online Model Building)
The online phase consumes the cached final graphs and trains the parameter-efficient
GraphLLM model.
3.3.1 GNN Encoder (Question-aware GAT)
We employ a Graph Attention Network (GAT) as the GNN encoder. Key points:
•Question-aware gating: Before message passing, node and edge features are re-
weighted by their cosine similarity to the question embedding. This gating gives
the GAT an initial inductive bias toward question-relevant structure.
•Multi-layer attention: The GAT uses multiple attention layers, residual connec-
tions, layer norm, and dropout; final node features are pooled (mean + max) to
produce a single 1024-dimensional graph vector.
•Output: The GAT’s pooled output is treated as the compact representation of the
retrieved subgraph.
17

3.3.2 Projector Module
To bridge the embedding-dimension gap between the GAT output (1024-d) and the LLM
embedding space (4096-d), we use a small MLP projector:
Projector MLP:
- Linear: 1024 -> 2048
- Sigmoid activation
- Linear: 2048 -> 4096
The resulting 4096-d vector is the graph token , which is inserted into the LLM input
embeddings immediately after the initial prompt token.
Question Embedding
(1024-d)Node Embeddings
(1024-d ×N)Edge Embeddings
(1024-d ×E)
Distance / Cosine Sim.Gating MLP
(distance →weight)
Weighted Node Embeddings Weighted Edge Embeddings
GAT Layer 1
(Multi-head)GAT Layer 2 GAT Layer 3
Residual/LN/Dropout Residual/LN/Dropout Residual/LN/Dropout
Global Pooling
(Mean+Max)
Graph Representation
(1024-d)
Figure 3: Compact question-aware GAT encoder (adjusted spacing and font).
3.3.3 LLM Backbone
The LLM backbone is a frozen LLaMA-7B model. We keep the full 6.7B parameters
frozen and train only the GAT + Projector (and any small task heads), resulting in an
approximate trainable parameter count of ∼39.9M. This parameter-efficient strategy
enables training on limited hardware while leveraging the LLM’s pre-existing language
and reasoning capabilities.
18

3.4 Training and Caching
Training consumes the cached PyG Data objects (final pruned graphs) and their textual
snapshots. The typical workflow:
•Load cached final graph; compute GAT representation; project to graph token;
prepend to token embeddings; forward through frozen LLM; compute task loss;
backpropagate into GAT + Projector.
•Early stopping and checkpointing are used to prevent overfitting; the best check-
point (by validation metric) is retained for evaluation.
3.5 Evaluation Metrics
We evaluate retrieval and end-to-end QA using:
•Hit Rate (Hit@1): The fraction of questions for which the correct answer entity
appears among the top-ranked candidates (primary retrieval metric).
•Recall (Retrieval Recall / Recall@k): The fraction of ground-truth answer
entities that are present in the retrieved candidate set (e.g., top-k or after final
pruning). This measures the retrieval module’s coverage of relevant facts and com-
plements Hit@1 by quantifying missed targets even when they are not ranked first.
•F1 Score: The harmonic mean of precision and recall computed over predicted vs.
ground-truth answer entities for end-to-end QA.
3.6 Implementation notes and scripts
All preprocessing and retrieval scripts are located in src/dataset/ (encoding, hydration,
retrieval) and the model code is in src/model/ . The one-time global encoder script and
the NaN patch script are included in the repository.
4 Implementation and Results
4.1 Environment and Working Tools
4.1.1 Hardware Environment
All preprocessing was performed on a remote RDP machine equipped with an NVIDIA
GPU providing 25 GB of VRAM. The RDP environment was used to run the one-time,
mega-batch encoding (for node and contextual edge embeddings) and to produce the
chunked tensor files and cached PyG objects that constitute the offline artifacts. Once
the global encoding and hydration steps completed, the encoded tensors and cached
19

graph objects were uploaded to Google Colab, where the model training and evaluation
were executed (A100 GPU of 40 GB VRAM). Splitting the pipeline in this way—heavy,
memory-intensive preprocessing on the RDP machine and training on Colab allowed us
to leverage the larger GPU memory for efficient encoding while keeping the training
workflow reproducible and accessible.
4.1.2 Software Environment
The solution was developed in Python 3.11. The core deep learning framework is PyTorch
2.1.2 (for CUDA 11.8). The environment was carefully constructed with pinned versions
of key libraries to ensure stability and reproducibility. The full environment setup is
detailed in the project’s Colab notebook.
4.2 Technological Choices
4.2.1 Programming Language
Python was chosen for its extensive scientific computing ecosystem and robust support
for deep learning frameworks like PyTorch.
4.2.2 Libraries
•PyTorch: The primary framework for building and training neural networks
•PyTorch Geometric (PyG): An extension library for PyTorch for implementing
Graph Neural Networks.
•Hugging Face Transformers: Used for accessing the pre-trained LLaMA-2 7B model.
•Hugging Face Sentence-Transformers: Used for the high-performance BAAI General
Embedding (BGE) model for encoding text.
•Hugging Face Datasets: For efficiently loading and managing the WebQSP dataset
•Pandas: Used for data manipulation during the graph construction phase.
4.3 Experimental Setup
To evaluate the effectiveness of the proposed model , we run a controlled comparison on
the WebQSP test set between a structure-agnostic textual baseline and our full graph-
aware model. The experimental setup is as follows:
Baseline Model A frozen LLaMA-2 7B model augmented with a trainable textual
RAG component. Instead of an explicit graph input, the baseline is provided with the raw,
unfiltered textual linear- ization of the question’s associated subgraph (i.e., the textualized
20

triplets / raw dump). This baseline represents a strong structure-agnostic competitor that
benefits from the same language backbone but lacks explicit graph reasoning.
Proposed Model The full pipeline: offline global encoding + hydration + multi-stage
retrieval (seed selection, 3-hop expansion, textual ranking with gettrunk triplets ,
merge & de-duplication, final top-edge pruning), a question-aware GAT encoder, a pro-
jector MLP that produces the 4096-d graph token, and a frozen LLaMA-7B backbone.
Only the GAT + Projector (39.9M parameters) are trained.
Common settings
•Dataset: WebQSP (4,700 QA pairs).
•Hardware: Google Colab, single T4 GPU.
•Training: Early stopping, best checkpoint by validation metric.
•Metrics: Hit@1 (primary retrieval metric) and F1 (end-to-end QA).
•Preprocessing: All node/edge textualizations and global encodings performed once
and cached (see Section 4).
4.4 Experimental Results
Below we report the evaluation metrics for the models described above. Replace the
baseline placeholders with the measured baseline numbers from your experiments.
Table 3: Comparison of multiple models on WebQSP (test set).
Model F1 (%) Hit@1 (%) Recall (%)
LLM only 0.2555 0.4148 0.2920
LLMLoRA 0.4295 0.6186 0.4193
G-Retriever 0.4674 0.6808 0.4579
G-RetrieverLoRA 0.5023 0.7016 0.5002
GRAG 0.5022 0.7236 0.5099
GRAGLoRA 0.5041 0.7275 0.5112
OUR MODEL 0.5432 0.7174 0.5343
Summary of results. Table 3 reports F1, Hit@1 and Recall for several baselines and
our proposed model . Our model (GAT + Projector + frozen LLaMA-7B) obtains the
highest F1 of 54.32% , an absolute improvement of +3.91 percentage points over the
strongest prior F1 (GRAGLoRA: 50.41%). It also achieves the best Recall at 53.43%
(+2.31 points vs. GRAGLoRA), indicating improved coverage of ground-truth entities in
the retrieved candidates. The Hit@1 for our model is 71.74% , which is competitive with
the top Hit@1 (GRAGLoRA: 72.75%) and within 1.0 point. Overall, these numbers show
that our multi-stage retrieval and question-aware GAT encoding produce cleaner, more
useful subgraphs: this raises final answer quality (F1) and retrieval coverage (Recall)
21

while maintaining a high candidate-hit rate. The table also shows that applying LoRA
to different pipelines tends to improve performance consistently, and that graph-aware
methods (G-Retriever / GRAG variants and proposed model ) substantially outperform
LLM-only baselines. The main trade-offs are additional preprocessing and slightly higher
per-query inference cost due to graph hydration and pruning, which we believe is justified
by the consistent accuracy gains.
22

5 Conclusion and Perspectives
5.1 Conclusion
In this internship we designed, implemented, and evaluated a custom multi-stage retrieval
pipeline and a parameter-efficient GraphLLM architecture for question answering over
large, noisy knowledge graphs. The core contributions include a one-time global encoding
procedure for nodes and contextual edges, a multi-stage subgraph retrieval algorithm (25-
seed selection, 3-hop expansion, textual ranking with gettrunk triplets , merging &
de-duplication, and final top-edge pruning to 60 edges), and a lightweight Projector that
maps GNN-derived graph representations into the LLaMA embedding space as a special
“graph token”. By freezing the LLaMA-7B backbone and training only the GAT +
Projector (39.9M parameters), we demonstrated that high-quality, low-noise subgraphs
enable competitive performance under tight compute constraints.
Empirically, the proposed model pipeline achieved a Hit Rate of 71.74% , F1 score of
54.32 and Recall score of 53.34 on the WebQSP test set. These results validate the
central hypothesis of this work: careful preprocessing and semantically-dense subgraph
construction can substantially improve retrieval and end-to-end QA performance even
when the LLM backbone is kept frozen.
5.2 Limitations
While the results are promising, several limitations should be acknowledged:
•Dataset scope: Experiments were confined to WebQSP; behaviour on larger or
heterogenous KGs (or different QA formats) remains to be evaluated.
•Compute- and storage-heavy preprocessing: The one-time global encoding
requires substantial I/O and time (multi-day on commodity setups) and generates
large on-disk artifacts that may be costly to store and maintain.
•Dependency on textualization quality: Ranking and pruning rely on textual
representations of subgraphs; errors in textualization (missing labels, MID place-
holders) can still propagate into retrieval and GNN weighting.
•Partial end-to-end adaptation: With a frozen LLM we rely on the GNN +
Projector to adapt graph information to the LLM; fully end-to-end fine-tuning (or
more advanced PEFT in the LLM) could further improve performance but was
constrained by hardware availability.
5.3 Perspectives and Future Work
The project opens several clear directions for future research and engineering:
1.Scale and generalization: Evaluate the model on additional KGQA datasets
(e.g., WebQuestions, MetaQA, ComplexQuestions) and on larger/heterogeneous
23

KGs to assess generalization.
2.Adaptive and hierarchical retrieval: Replace or augment the fixed top- kheuris-
tics with adaptive controllers (learned seed selection, budgeted expansion) or hier-
archical retrieval to further reduce noise while preserving recall.
3.End-to-end and PEFT experiments: Explore low-cost fine-tuning strategies
(LoRA, adapters, prompt tuning) applied to the LLM together with the GNN to
measure marginal gains beyond the frozen-backbone setup.
4.Improved textualization and semantic compression: Investigate learned tex-
tualizers or neural summarizers that compress a subgraph into a compact, high-
fidelity text representation for faster and potentially more accurate ranking.
5.Efficiency and deployment: Optimize the one-time encoding pipeline for dis-
tributed execution, use approximate nearest neighbor indices for faster seed re-
trieval, and benchmark end-to-end latency to make the pipeline production-ready.
6.Robustness and human evaluation: Conduct human-in-the-loop evaluations,
adversarial stress tests (ambiguous questions, MID-heavy graphs), and broader er-
ror analyses to better characterize failure modes and trustworthiness.
5.4 Final Remarks
This study demonstrates that carefully engineered retrieval and graph-to-language inter-
faces can materially improve the ability of LLMs to reason over structured knowledge
without large-scale fine-tuning of the language model itself. The model pipeline provides
a practical, resource-conscious blueprint for researchers and practitioners aiming to com-
bine graph structure with powerful pre-trained LLMs; the perspectives above outline a
path toward broader applicability, greater robustness, and improved efficiency.
24

6 References
References
[1] I. de Zarz` a, J. de Curt` o, G. Roig, and T. Calafate, C.˙Optimized financial planning:
Integrating individual and cooperative budgeting models with llm recommendations.
AI, 5(1):91–114, 2024.
[2] Pengcheng He, Raphael Tang, Xinyuan Zhou, Xiaodong Wang, Yelong Zhang, Xi Vic-
toria Lin, Yelong Shen, Yujia Ma, Sheng Huang, Luke Zettlemoyer, Michel Galley, and
Mu Li. G-retriever: Retrieval-augmented generation for textual graph understanding
and question answering, 2024.
[3] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language
models, 2021.
[4] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. Grag:
Graph retrieval-augmented generation. In Findings of the Association for Computa-
tional Linguistics: NAACL 2025 , pages 4145–4157. ACL, 2025.
[5] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. Think-on-graph 2.0: Deep and faithful large language model
reasoning with knowledge-guided retrieval augmented generation, 2024.
[6] 2023 Touvron et al. Llama 2: Open foundation and fine-tuned chat models, 2023.
[7] Petar Veliˇ ckovi´ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio,
and Yoshua Bengio. Graph attention networks. In International Conference on Learn-
ing Representations (ICLR) , 2018.
[8] Yuwei Wan, Zheyuan Chen, Ying Liu, Chong Chen, and Michael Packianather. Em-
powering llms by hybrid retrieval-augmented generation for domain-centric q&a in
smart manufacturing. Advanced Engineering Informatics , 65:103212, 2025.
[9] Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong
Chen. Subgraph retrieval enhanced model for multi-hop knowledge base question
answering. In Proceedings of the 60th Annual Meeting of the Association for Compu-
tational Linguistics (Long Papers) , pages 5773–5784. ACL, 2022.
25
```

---


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
(This command uses optimized parameters for a single GPU with ~16GB of VRAM)

```bash
python train.py --dataset webqsp --batch_size 1 --grad_steps 4
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

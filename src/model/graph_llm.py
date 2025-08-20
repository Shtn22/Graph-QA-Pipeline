import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = { "device_map": "auto", "revision": "main" }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for _, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r, lora_alpha, lora_dropout = 8, 16, 0.05
            lora_target_modules = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules,
                lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim, out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim, num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout, mlp_layers = args.alignment_mlp_layers,
            num_heads=args.gnn_num_heads, operator=args.distance_operator,
        ).to(self.model.device)
        
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)
        if graphs.num_nodes == 0:
            return torch.empty(0, self.projector[0].in_features, device=self.device)
        n_embeds, _ = self.graph_encoder(graphs.x, 
                                         graphs.edge_index.long(), 
                                         graphs.question_node,
                                         graphs.edge_attr, 
                                         graphs.question_edge)
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        return g_embeds

    def forward(self, samples):
        if samples['graph'].num_nodes == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        
        graph_embeds = self.projector(self.encode_graphs(samples))
        
        batch_size = len(samples['id'])
        batch_inputs_embeds, batch_attention_mask, batch_label_input_ids = [], [], []
        
        graph_embeds_idx = 0
        for i in range(batch_size):
            if samples['graph'][i].num_nodes > 0:
                label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
                inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[graph_embeds_idx].unsqueeze(0), inputs_embeds], dim=0)
                graph_embeds_idx += 1
                batch_inputs_embeds.append(inputs_embeds)
                batch_attention_mask.append([1] * inputs_embeds.shape[0])
                label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
                batch_label_input_ids.append(label_input_ids)

        if not batch_inputs_embeds:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(len(batch_inputs_embeds)):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    def inference(self, samples):
        if samples['graph'].num_nodes == 0:
            return {'id': samples['id'], 'pred': [""] * len(samples['id']), 'label': samples['label'],
                    'question': samples['question'], 'desc': samples['desc'], }

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        
        graph_embeds = self.projector(self.encode_graphs(samples))
        
        batch_size = len(samples['id'])
        batch_inputs_embeds, batch_attention_mask = [], []
        graph_embeds_idx = 0
        for i in range(batch_size):
            if samples['graph'][i].num_nodes > 0:
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
                inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[graph_embeds_idx].unsqueeze(0), inputs_embeds], dim=0)
                graph_embeds_idx += 1
                batch_inputs_embeds.append(inputs_embeds)
                batch_attention_mask.append([1] * inputs_embeds.shape[0])
            
        if not batch_inputs_embeds:
            return {'id': samples['id'], 'pred': [""] * len(samples['id']), 'label': samples['label'],
                    'question': samples['question'], 'desc': samples['desc'], }
            
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        
        non_empty_indices = [i for i in range(batch_size) if samples['graph'][i].num_nodes > 0]
        for i in range(len(batch_inputs_embeds)):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True
            )
        pred_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        full_pred = [""] * batch_size
        for i, original_index in enumerate(non_empty_indices):
            full_pred[original_index] = pred_texts[i]
            
        return {'id': samples['id'], 'pred': full_pred, 'label': samples['label'],
                'question': samples['question'], 'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params, all_param = 0, 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
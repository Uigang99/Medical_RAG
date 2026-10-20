"""Frozen-Llama diagnostic: open future keys ONLY across existing KG proposition pairs.

Input rows remain the original causal layouts. Reverse directions are derived,
not written into those caches. Direction-specific biases are independent.
Question/options/instructions/padding are never endpoints of opened edges.
This wrapper scores the next MCQ label, not a rationale or cached generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from medrag.generation.proposition_kg_attention import (
    NAME, _AttentionContext, PropositionAttentionModel, PropositionAttentionOutput,
    validate_layout, block_bias,
)
from medrag.training.attention_oracle import OracleBiasTable, oracle_attention
from medrag.training.attention_intervention_diagnostic import GROUPS


@dataclass(frozen=True)
class GraphAttentionContext(_AttentionContext):
    reverse_links: tuple[torch.Tensor, ...] = ()
    allowed_mask: torch.Tensor | None = None


def allowed_chunk(context, start, stop, device):
    if context.allowed_mask is not None:
        return context.allowed_mask[:, :, start:stop]
    n = len(context.token_propositions[0])
    keys = torch.arange(n, device=device)
    causal = keys[None, :] <= torch.arange(start, stop, device=device)[:, None]
    allowed = causal[None].expand(len(context.lengths), -1, -1)
    if context.reverse_links:
        links = torch.stack([adj[t[start:stop, None], t[None, :]]
                             for adj,t in zip(context.reverse_links,context.token_propositions,strict=True)])
        allowed = allowed | links
    valid = keys[None, :] < context.lengths[:, None]
    return (allowed & valid[:, None, :])[:, None]


def graph_attention(module, query, key, value, attention_mask, scaling, **kwargs):
    contexts = kwargs.get('zero_reference_contexts')
    context = contexts[module.layer_idx] if contexts is not None else kwargs.get('proposition_context')
    if not isinstance(context, GraphAttentionContext) or not context.reverse_links:
        return oracle_attention(module,query,key,value,attention_mask,scaling,**kwargs)
    if kwargs.get('dropout',0.): raise RuntimeError('Frozen target dropout must remain zero')
    if query.shape[-2] != key.shape[-2]: raise RuntimeError('Full-prompt diagnostic only; no KV decoding')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    out=[]
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0,query.shape[-2],context.query_chunk):
            stop=min(start+context.query_chunk,query.shape[-2])
            allowed=allowed_chunk(context,start,stop,query.device)
            mask=block_bias(context,start,stop,query.dtype).masked_fill(~allowed,-torch.inf)
            out.append(F.scaled_dot_product_attention(query[:,:,start:stop],key,value,
                attn_mask=mask,dropout_p=0.,is_causal=False,scale=scaling,
                enable_gqa=query.shape[1]!=key.shape[1]))
    return torch.cat(out,2).transpose(1,2).contiguous(),None


def install():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(NAME,graph_attention)


class BidirectionalPropositionModel(PropositionAttentionModel):
    """Single-question oracle wrapper. Llama remains frozen; only tables change."""
    def __init__(self,llm,row,query_chunk=128,cache_prefix=False):
        super().__init__(llm,OracleBiasTable(llm.config.hidden_size,row['pairs']).to(llm.device),
            attention_layers=tuple(i for g in GROUPS for i in g),
            gradient_checkpointing=True,query_chunk=query_chunk)
        self.cache_prefix=cache_prefix
        self._prefix_key=self._prefix_value=None
        self._mask_key=self._mask_value=None
        self.prefix_computations=self.prefix_cache_hits=0
        self.runtime_profiles={}
        self.configure(row,False)
        install()

    def configure(self,row,bidirectional):
        validate_layout(row)
        self.bidirectional=bool(bidirectional)
        self.ordered_pairs=[list(p) for p in row['pairs']]
        if bidirectional:
            self.ordered_pairs += [[s,r] for r,s in row['pairs']]
        self.controller=OracleBiasTable(self.llm.config.hidden_size,self.ordered_pairs).to(self.llm.device)
        profile=self.runtime_profiles.get('bidirectional' if bidirectional else 'causal')
        if profile is not None:
            # An OOM fallback is local to one question, not sticky for all later questions.
            self.gradient_checkpointing=profile['gradient_checkpointing']
        self.eval()

    @torch.no_grad()
    def prefix(self,ids,positions,ordinary,input_key):
        # These layers precede EVERY learned bias and contain frozen weights only.
        # Key by exact input, not sample ID: altered text must invalidate this cache.
        key=(input_key,str(ids.device),self.cut_layer)
        if self.cache_prefix and key==self._prefix_key:
            self.prefix_cache_hits+=1
            return self._prefix_value
        hidden=self.llm.model.embed_tokens(ids)
        embeddings=self.llm.model.rotary_emb(hidden,position_ids=positions)
        for layer in self.llm.model.layers[:self.cut_layer]:
            hidden=layer(hidden,position_ids=positions,position_embeddings=embeddings,
                         use_cache=False,proposition_context=ordinary)
        self.prefix_computations+=1
        if self.cache_prefix:
            self._prefix_key,self._prefix_value=key,(hidden,embeddings)
        return hidden,embeddings

    def forward(self,rows,pad_id):
        if len(rows)!=1: raise ValueError('Question-local oracle uses batch size 1')
        row=rows[0];validate_layout(row)
        expected=row['pairs']+([[s,r] for r,s in row['pairs']] if self.bidirectional else [])
        if expected!=self.ordered_pairs: raise RuntimeError('Configure the oracle table for this row first')
        device=self.llm.device; n=len(row['input_ids']); spans=row['proposition_spans']
        if n>self.llm.config.max_position_embeddings: raise ValueError('Prompt exceeds model context')
        ids=torch.tensor([row['input_ids']],device=device)
        token_props=torch.full((n,),len(spans),device=device,dtype=torch.long)
        for i,(a,b) in enumerate(spans): token_props[a:b]=i
        lengths=torch.tensor([n],device=device); positions=torch.arange(n,device=device)[None]
        ordinary=_AttentionContext(lengths,(token_props,),None,self.query_chunk)
        hidden,embeddings=self.prefix(ids,positions,ordinary,tuple(row['input_ids']))
        pairs=self.controller.expected_pairs
        bias=self.controller(None,None,pairs,None)
        size=len(spans)+1
        links=torch.zeros((size,size),device=device,dtype=torch.bool)
        if self.bidirectional and len(pairs): links[pairs[:,0],pairs[:,1]]=True
        allowed=None
        if self.cache_prefix and self.bidirectional and len(pairs):
            key=(tuple(map(tuple,spans)),n,tuple(map(tuple,self.ordered_pairs)),str(device))
            if key!=self._mask_key:
                c=GraphAttentionContext(lengths,(token_props,),None,self.query_chunk,(),(links,))
                self._mask_key,self._mask_value=key,allowed_chunk(c,0,n,device)
            allowed=self._mask_value
        readers=(tuple(tuple(spans[i]) for i in sorted({p[0] for p in self.ordered_pairs})),)
        contexts={}
        for col,layer_index in enumerate(self.attention_layers):
            matrix=bias.new_zeros((size,size))
            if len(pairs): matrix=matrix.index_put((pairs[:,0],pairs[:,1]),bias[:,col])
            contexts[layer_index]=GraphAttentionContext(lengths,(token_props,),(matrix,),self.query_chunk,
                readers,(links,) if self.bidirectional and len(pairs) else (),allowed)
        for index in range(self.cut_layer,len(self.llm.model.layers)):
            layer=self.llm.model.layers[index];context=contexts.get(index,ordinary)
            def execute(value,layer=layer,context=context):
                return layer(value,position_ids=positions,position_embeddings=embeddings,
                             use_cache=False,proposition_context=context)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden=checkpoint(execute,hidden,use_reentrant=False,preserve_rng_state=False)
            else: hidden=execute(hidden)
        last=self.llm.model.norm(hidden[:,-1])
        penalty=bias.square().mean() if bias.numel() else bias.sum()*0.
        return PropositionAttentionOutput(last,penalty[None],(bias,),dict(
            candidate_directions=len(self.ordered_pairs),bidirectional=self.bidirectional))

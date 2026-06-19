"""BQ (Bisimulation Quotienting) policy network for VNE.

Single unified transformer encoder processes ALL tokens (substrate nodes, edges,
virtual requests, candidate paths) in one flat sequence — no encoder/decoder split.

Architecture follows the paper "BQ-NCO: Bisimulated Quotienting for Efficient
Neural Combinatorial Optimization" (Drakulic et al.), as used in the TaSaR paper
for TSP (9 blocks, dim=128) and CVRP (9 blocks, dim=192).

Key difference from the LEHD network (vne/network.py):
  - LEHD: separate graph encoder → decoder with [global, virtual, candidates]
  - BQ:   single transformer stack over [nodes | edges | virtuals | candidates]
          with all-to-all attention, then linear projection from candidate positions
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from modules.rztx import RZTXEncoderLayer
from tsp.lehd_network import NoNormTransformerEncoderLayer, dict_to_cpu
from vne.features import (
    CANDIDATE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    VIRTUAL_FEATURE_DIM,
)


class BQPolicyNetwork(nn.Module):
    """BQ-style VNE policy network.

    The action space is the same as LEHD: each output logit scores one feasible
    candidate substrate path for the current virtual link.
    """

    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        latent_dim = getattr(config, "latent_dimension", config.embedding_dim)
        feedforward_dim = getattr(config, "feedforward_dimension", 4 * latent_dim)
        num_heads = getattr(config, "num_attention_heads", 8)
        dropout = getattr(config, "dropout", 0.0)
        num_blocks = getattr(config, "num_transformer_blocks", 9)
        use_rezero = getattr(config, "use_rezero_transformer", True)

        # Four token types: node (0), edge (1), virtual (2), candidate (3)
        self.node_embedding = nn.Linear(NODE_FEATURE_DIM, latent_dim)
        self.edge_embedding = nn.Linear(EDGE_FEATURE_DIM, latent_dim)
        self.virtual_embedding = nn.Linear(VIRTUAL_FEATURE_DIM, latent_dim)
        self.candidate_embedding = nn.Linear(CANDIDATE_FEATURE_DIM, latent_dim)
        self.token_type_embedding = nn.Embedding(4, latent_dim)

        # Single unified transformer stack (matching paper's BQ: 9 blocks)
        block_cls = RZTXEncoderLayer if use_rezero else NoNormTransformerEncoderLayer
        self.graph_encoder = nn.ModuleList(
            [
                block_cls(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )

        # Single linear projection from latent to one logit per candidate
        self.policy_linear_out = nn.Linear(latent_dim, 1)

    # ------------------------------------------------------------------
    # Padding helpers (same pattern as VNEPolicyNetwork)
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_2d(tensors: List[torch.Tensor], max_len: int, feat_dim: int):
        """Stack a list of [n_i, F] into [B, max_len, F] (zero-pad) + bool pad mask [B, max_len]."""
        batch = len(tensors)
        out = torch.zeros((batch, max_len, feat_dim), dtype=torch.float32)
        pad = torch.ones((batch, max_len), dtype=torch.bool)
        for i, t in enumerate(tensors):
            n = t.shape[0]
            if n:
                out[i, :n] = t
                pad[i, :n] = False
        return out, pad

    # ------------------------------------------------------------------
    # Batched forward
    # ------------------------------------------------------------------

    def _forward_batch(self, states: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """Batched forward over a list of states.

        All token types (nodes, edges, virtuals, candidates) are padded to their
        respective max lengths, concatenated into ONE flat sequence per batch element,
        and processed through the unified transformer stack. Candidate positions are
        then extracted and projected to logits.

        This is the defining BQ pattern: no separate encoder/decoder, no cross-attention,
        just one transformer stack over everything.
        """
        device = self.device
        latent = self.policy_linear_out.in_features
        batch = len(states)

        # Token counts per instance
        node_counts = [int(s["node_features"].shape[0]) for s in states]
        edge_counts = [int(s["edge_features"].shape[0]) for s in states]
        virt_counts = [int(s["virtual_features"].shape[0]) for s in states]
        cand_counts = [int(s["candidate_features"].shape[0]) for s in states]
        max_n, max_e, max_v = max(node_counts), max(edge_counts), max(virt_counts)
        max_c = max(cand_counts)

        if max_c == 0:
            return [torch.empty((0,), device=device) for _ in states]

        # Pad raw features on CPU, then single move to device
        node_feat, node_pad = self._pad_2d(
            [s["node_features"] for s in states], max_n, NODE_FEATURE_DIM
        )
        edge_feat, edge_pad = self._pad_2d(
            [s["edge_features"] for s in states], max_e, EDGE_FEATURE_DIM
        )
        virt_feat, virt_pad = self._pad_2d(
            [s["virtual_features"] for s in states], max_v, VIRTUAL_FEATURE_DIM
        )
        cand_feat, cand_pad = self._pad_2d(
            [s["candidate_features"] for s in states], max_c, CANDIDATE_FEATURE_DIM
        )

        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)
        virt_feat = virt_feat.to(device)
        cand_feat = cand_feat.to(device)
        node_pad = node_pad.to(device)
        edge_pad = edge_pad.to(device)
        virt_pad = virt_pad.to(device)
        cand_pad = cand_pad.to(device)

        # Embed and add type markers
        type_w = self.token_type_embedding.weight  # [4, D]
        node_tok = self.node_embedding(node_feat) + type_w[0].view(1, 1, -1)
        edge_tok = self.edge_embedding(edge_feat) + type_w[1].view(1, 1, -1)
        virt_tok = self.virtual_embedding(virt_feat) + type_w[2].view(1, 1, -1)
        cand_tok = self.candidate_embedding(cand_feat) + type_w[3].view(1, 1, -1)

        # Concatenate into single flat sequence: [nodes | edges | virtuals | candidates]
        seq = torch.cat([node_tok, edge_tok, virt_tok, cand_tok], dim=1)
        pad_mask = torch.cat([node_pad, edge_pad, virt_pad, cand_pad], dim=1)

        # Unified transformer stack — all tokens attend to all others
        for block in self.graph_encoder:
            seq = block(seq, src_key_padding_mask=pad_mask)

        # Extract candidate positions and project to logits
        cand_start = max_n + max_e + max_v
        cand_outputs = seq[:, cand_start: cand_start + max_c, :]  # [B, max_c, D]
        logits_all = self.policy_linear_out(cand_outputs).squeeze(-1)  # [B, max_c]

        return [logits_all[i, : cand_counts[i]] for i in range(batch)]

    # ------------------------------------------------------------------
    # Sub-batching (same budget logic as VNEPolicyNetwork)
    # ------------------------------------------------------------------

    # Sub-batching budget: cap on (sub_batch_size * total_seq_len) per forward pass.
    # At 60-80n scale each instance has ~70+70+52+cand ≈ 192+cand tokens.
    # With budget=2400 and 400 total tokens: ~6 inst/sub-batch → ~60 MB attn per layer.
    total_token_budget = 2400

    def forward(self, state_batch: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        if not state_batch:
            return []
        results: List[torch.Tensor] = [None] * len(state_batch)
        # Sort by total token count (not just candidates) to keep sub-batches tight
        def _total_tokens(idx: int) -> int:
            s = state_batch[idx]
            return (
                int(s["node_features"].shape[0])
                + int(s["edge_features"].shape[0])
                + int(s["virtual_features"].shape[0])
                + int(s["candidate_features"].shape[0])
            )
        order = sorted(range(len(state_batch)), key=_total_tokens)
        i = 0
        while i < len(order):
            group: List[int] = []
            group_max = 0
            while i < len(order):
                tokens = max(1, _total_tokens(order[i]))
                new_max = max(group_max, tokens)
                if group and (len(group) + 1) * new_max > self.total_token_budget:
                    break
                group.append(order[i])
                group_max = new_max
                i += 1
            sub_states = [state_batch[k] for k in group]
            for k, logits in zip(group, self._forward_batch(sub_states)):
                results[k] = logits
        return results

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

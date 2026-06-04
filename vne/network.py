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


class VNEPolicyNetwork(nn.Module):
    """LEHD-style VNE policy network.

    The action space remains unchanged: each output logit scores one feasible
    candidate substrate path for the current virtual link.
    """

    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        latent_dim = getattr(config, "latent_dimension", config.embedding_dim)
        feedforward_dim = getattr(config, "feedforward_dimension", 4 * latent_dim)
        num_heads = getattr(config, "num_attention_heads", 4)
        dropout = getattr(config, "dropout", 0.0)
        num_encoder_layers = getattr(config, "num_encoder_layers", 1)
        num_decoder_layers = getattr(config, "num_decoder_layers", 3)
        use_rezero = getattr(config, "use_rezero_transformer", False)

        self.node_embedding = nn.Linear(NODE_FEATURE_DIM, latent_dim)
        self.edge_embedding = nn.Linear(EDGE_FEATURE_DIM, latent_dim)
        self.virtual_embedding = nn.Linear(VIRTUAL_FEATURE_DIM, latent_dim)
        self.candidate_embedding = nn.Linear(CANDIDATE_FEATURE_DIM, latent_dim)
        self.token_type_embedding = nn.Embedding(3, latent_dim)

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
                for _ in range(num_encoder_layers)
            ]
        )
        self.graph_decoder = nn.ModuleList(
            [
                block_cls(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.global_projection = nn.Linear(latent_dim, latent_dim)
        self.current_virtual_projection = nn.Linear(latent_dim, latent_dim)
        self.candidate_projection = nn.Linear(latent_dim, latent_dim)
        self.policy_linear_out = nn.Linear(latent_dim, 1)

    def _move_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in state.items()
        }

    def _encode_state(self, state: Dict[str, torch.Tensor]):
        node_tokens = self.node_embedding(state["node_features"])
        edge_tokens = self.edge_embedding(state["edge_features"])
        virtual_tokens = self.virtual_embedding(state["virtual_features"])

        token_groups = []
        if node_tokens.shape[0] > 0:
            token_groups.append(
                node_tokens + self.token_type_embedding.weight[0].view(1, -1)
            )
        if edge_tokens.shape[0] > 0:
            token_groups.append(
                edge_tokens + self.token_type_embedding.weight[1].view(1, -1)
            )
        if virtual_tokens.shape[0] > 0:
            token_groups.append(
                virtual_tokens + self.token_type_embedding.weight[2].view(1, -1)
            )

        encoded = torch.cat(token_groups, dim=0).unsqueeze(0)
        for trf_block in self.graph_encoder:
            encoded = trf_block(encoded)
        encoded = encoded.squeeze(0)

        node_count = node_tokens.shape[0]
        edge_count = edge_tokens.shape[0]
        virtual_count = virtual_tokens.shape[0]
        encoded_nodes = encoded[:node_count]
        encoded_edges = encoded[node_count: node_count + edge_count]
        encoded_virtual = encoded[
            node_count + edge_count: node_count + edge_count + virtual_count
        ]
        return encoded, encoded_nodes, encoded_edges, encoded_virtual

    def _pool_indexed_tokens(
        self,
        encoded_tokens: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        num_candidates = indices.shape[0]
        latent_dim = self.policy_linear_out.in_features
        if num_candidates == 0:
            return torch.zeros((0, latent_dim), device=self.device)
        if encoded_tokens.shape[0] == 0:
            return torch.zeros((num_candidates, latent_dim), device=self.device)

        safe_indices = indices.clamp(min=0, max=encoded_tokens.shape[0] - 1)
        gathered = encoded_tokens[safe_indices]
        valid = mask.to(self.device).unsqueeze(-1).float()
        denominator = valid.sum(dim=1).clamp_min(1.0)
        return (gathered * valid).sum(dim=1) / denominator

    def _forward_one(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = self._move_state(state)
        candidate_features = state["candidate_features"]
        if candidate_features.numel() == 0:
            return torch.empty((0,), device=self.device)

        encoded, encoded_nodes, encoded_edges, encoded_virtual = self._encode_state(state)
        global_context = self.global_projection(encoded.mean(dim=0, keepdim=True))
        current_virtual = encoded_virtual[state["current_virtual_index"].item()].view(1, -1)
        current_virtual = self.current_virtual_projection(current_virtual)

        candidate_base = self.candidate_embedding(candidate_features)
        candidate_node_pool = self._pool_indexed_tokens(
            encoded_nodes,
            state["candidate_node_indices"],
            state["candidate_node_mask"],
        )
        candidate_edge_pool = self._pool_indexed_tokens(
            encoded_edges,
            state["candidate_edge_indices"],
            state["candidate_edge_mask"],
        )
        candidate_tokens = self.candidate_projection(
            candidate_base + candidate_node_pool + candidate_edge_pool
        )

        seq = torch.cat(
            [
                global_context,
                current_virtual,
                candidate_tokens,
            ],
            dim=0,
        ).unsqueeze(0)
        for trf_block in self.graph_decoder:
            seq = trf_block(seq)

        logits = self.policy_linear_out(seq.squeeze(0)[2:]).squeeze(-1)
        return logits

    def forward(self, state_batch: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        return [self._forward_one(state) for state in state_batch]

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

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

    @staticmethod
    def _pad_2d(tensors: List[torch.Tensor], max_len: int, feat_dim: int):
        """Stack a list of [n_i, F] into [B, max_len, F] (zero-pad) + bool pad mask [B, max_len]."""
        batch = len(tensors)
        out = torch.zeros((batch, max_len, feat_dim), dtype=torch.float32)
        pad = torch.ones((batch, max_len), dtype=torch.bool)  # True == padding
        for i, t in enumerate(tensors):
            n = t.shape[0]
            if n:
                out[i, :n] = t
                pad[i, :n] = False
        return out, pad

    @staticmethod
    def _pad_3d_long(tensors: List[torch.Tensor], max_rows: int, max_cols: int, dtype):
        """Stack a list of [c_i, l_i] into [B, max_rows, max_cols] (zero-pad)."""
        batch = len(tensors)
        out = torch.zeros((batch, max_rows, max_cols), dtype=dtype)
        for i, t in enumerate(tensors):
            r, c = t.shape
            if r and c:
                out[i, :r, :c] = t
        return out

    def _forward_batch(self, states: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """Batched equivalent of `_forward_one` over a list of states.

        The encoder/decoder carry no positional encoding (pure set attention), so
        block-padding the variable-length node/edge/virtual/candidate sequences and
        masking the pads yields exactly the same per-candidate logits as processing
        each state alone -- but in one batched pass with a single host->device copy.
        """
        device = self.device
        latent = self.policy_linear_out.in_features
        batch = len(states)

        node_counts = [int(s["node_features"].shape[0]) for s in states]
        edge_counts = [int(s["edge_features"].shape[0]) for s in states]
        virt_counts = [int(s["virtual_features"].shape[0]) for s in states]
        cand_counts = [int(s["candidate_features"].shape[0]) for s in states]
        max_n, max_e, max_v = max(node_counts), max(edge_counts), max(virt_counts)
        max_c = max(cand_counts)

        if max_c == 0:
            return [torch.empty((0,), device=device) for _ in states]

        # --- Pad raw features (CPU), then a single move to device. ---
        node_feat, node_pad = self._pad_2d([s["node_features"] for s in states], max_n, NODE_FEATURE_DIM)
        edge_feat, edge_pad = self._pad_2d([s["edge_features"] for s in states], max_e, EDGE_FEATURE_DIM)
        virt_feat, virt_pad = self._pad_2d([s["virtual_features"] for s in states], max_v, VIRTUAL_FEATURE_DIM)
        cand_feat, cand_pad = self._pad_2d([s["candidate_features"] for s in states], max_c, CANDIDATE_FEATURE_DIM)

        max_pn = max(int(s["candidate_node_indices"].shape[1]) for s in states)
        max_pe = max(int(s["candidate_edge_indices"].shape[1]) for s in states)
        cand_node_idx = self._pad_3d_long([s["candidate_node_indices"] for s in states], max_c, max_pn, torch.long)
        cand_node_msk = self._pad_3d_long([s["candidate_node_mask"] for s in states], max_c, max_pn, torch.bool)
        cand_edge_idx = self._pad_3d_long([s["candidate_edge_indices"] for s in states], max_c, max_pe, torch.long)
        cand_edge_msk = self._pad_3d_long([s["candidate_edge_mask"] for s in states], max_c, max_pe, torch.bool)
        cur_virtual_idx = torch.tensor([int(s["current_virtual_index"]) for s in states], dtype=torch.long)

        node_feat = node_feat.to(device); edge_feat = edge_feat.to(device); virt_feat = virt_feat.to(device)
        cand_feat = cand_feat.to(device)
        node_pad = node_pad.to(device); edge_pad = edge_pad.to(device); virt_pad = virt_pad.to(device)
        cand_pad = cand_pad.to(device)
        cand_node_idx = cand_node_idx.to(device); cand_node_msk = cand_node_msk.to(device)
        cand_edge_idx = cand_edge_idx.to(device); cand_edge_msk = cand_edge_msk.to(device)
        cur_virtual_idx = cur_virtual_idx.to(device)

        type_w = self.token_type_embedding.weight  # [3, D]
        node_tok = self.node_embedding(node_feat) + type_w[0].view(1, 1, -1)
        edge_tok = self.edge_embedding(edge_feat) + type_w[1].view(1, 1, -1)
        virt_tok = self.virtual_embedding(virt_feat) + type_w[2].view(1, 1, -1)

        # Encoder over [nodes | edges | virtual] blocks (order is irrelevant: no
        # positional encoding), with a key-padding mask removing pad slots.
        enc = torch.cat([node_tok, edge_tok, virt_tok], dim=1)
        enc_pad = torch.cat([node_pad, edge_pad, virt_pad], dim=1)
        for block in self.graph_encoder:
            enc = block(enc, src_key_padding_mask=enc_pad)

        enc_nodes = enc[:, :max_n]
        enc_virt = enc[:, max_n + max_e: max_n + max_e + max_v]

        # Global context = mean over the real tokens only.
        real = (~enc_pad).float().unsqueeze(-1)
        global_ctx = (enc * real).sum(dim=1) / real.sum(dim=1).clamp_min(1.0)
        global_ctx = self.global_projection(global_ctx)  # [B, D]

        current_virtual = enc_virt[torch.arange(batch, device=device), cur_virtual_idx]
        current_virtual = self.current_virtual_projection(current_virtual)  # [B, D]

        # Candidate path pooling: gather encoded node/edge tokens along each path.
        enc_edges = enc[:, max_n: max_n + max_e]
        node_pool = self._gather_pool(enc_nodes, cand_node_idx, cand_node_msk, latent)
        edge_pool = self._gather_pool(enc_edges, cand_edge_idx, cand_edge_msk, latent)
        cand_base = self.candidate_embedding(cand_feat)
        cand_tokens = self.candidate_projection(cand_base + node_pool + edge_pool)  # [B, max_c, D]

        # Decoder over [global, current_virtual, candidates...] with pad mask on candidates.
        dec = torch.cat([global_ctx.unsqueeze(1), current_virtual.unsqueeze(1), cand_tokens], dim=1)
        prefix_pad = torch.zeros((batch, 2), dtype=torch.bool, device=device)
        dec_pad = torch.cat([prefix_pad, cand_pad], dim=1)
        for block in self.graph_decoder:
            dec = block(dec, src_key_padding_mask=dec_pad)

        logits_all = self.policy_linear_out(dec[:, 2:, :]).squeeze(-1)  # [B, max_c]
        return [logits_all[i, : cand_counts[i]] for i in range(batch)]

    def _gather_pool(self, encoded: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, latent: int) -> torch.Tensor:
        """Mean-pool encoded tokens gathered by per-candidate path indices.

        encoded [B, T, D]; idx/mask [B, C, L] -> pooled [B, C, D]."""
        batch, max_c, path_len = idx.shape
        token_count = encoded.shape[1]
        safe_idx = idx.clamp(min=0, max=max(token_count - 1, 0))
        flat = safe_idx.reshape(batch, max_c * path_len)
        gathered = torch.gather(
            encoded, 1, flat.unsqueeze(-1).expand(-1, -1, latent)
        ).reshape(batch, max_c, path_len, latent)
        valid = mask.unsqueeze(-1).float()
        return (gathered * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)

    # Sub-batching budget: cap on (sub_batch_size * max_decoder_tokens) per pass.
    # Decoder tokens = 1 (context) + virtuals + candidates.
    # Budget scales inversely with embedding dim (attention memory ∝ dim²).
    # Reference: dim=128 → budget=1200, dim=192 → budget=800, dim=256 → budget=600.
    @property
    def total_token_budget(self) -> int:
        return max(150, int(800 * (128 / self.config.embedding_dim) ** 2))

    def forward(self, state_batch: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        if not state_batch:
            return []
        results: List[torch.Tensor] = [None] * len(state_batch)  # type: ignore
        # Sort by total decoder token count (not just candidates) to keep sub-batches tight
        def _decoder_tokens(idx: int) -> int:
            s = state_batch[idx]
            return (
                1  # context
                + int(s["virtual_features"].shape[0])
                + int(s["candidate_features"].shape[0])
            )
        order = sorted(range(len(state_batch)), key=_decoder_tokens)
        i = 0
        while i < len(order):
            group: List[int] = []
            group_max = 0
            while i < len(order):
                tokens = max(1, _decoder_tokens(order[i]))
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

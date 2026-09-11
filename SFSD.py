# file: SFSD.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torchcrf import CRF


# ---------------------------------------------------------------------------
# Transformer 基础组件
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, drop_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.norm_in = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=0.1 if i > 0 else 0.0
            ) for i in range(num_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        return x


# ---------------------------------------------------------------------------
# MoE 模块
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class SparseDispatcher(nn.Module):
    def __init__(self, num_experts, gates):
        super().__init__()
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


class MLPExpert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.out_drop = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.out_drop(out)
        out = self.fc2(out)
        return out


class MoE_Block(nn.Module):
    """Top-k 路由 MoE 模块，带可导负载均衡损失和路由统计。"""

    def __init__(self, input_size, output_size, num_experts, hidden_size,
                 k=2, router_noise=0.01):
        super().__init__()
        self.num_experts = int(num_experts)
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = int(k)
        self.router_noise = float(router_noise)

        if not 1 <= self.k <= self.num_experts:
            raise ValueError(
                f"moe_top_k必须位于[1, num_experts]，当前k={self.k}, "
                f"num_experts={self.num_experts}"
            )

        self.experts = nn.ModuleList([
            MLPExpert(self.input_size, self.output_size, self.hidden_size)
            for _ in range(self.num_experts)
        ])

        # 不能使用全零初始化，否则Top-k在训练初期容易固定选择同一批专家。
        self.w_gate = nn.Parameter(torch.empty(input_size, self.num_experts))
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)

        self.rmsnorm = RMSNorm(dim=self.output_size)
        self.act = nn.GELU()

        self.last_importance = None
        self.last_load = None
        self.last_selection_share = None
        self.last_router_entropy = None

    def reset_routing_stats(self):
        self.last_importance = None
        self.last_load = None
        self.last_selection_share = None
        self.last_router_entropy = None

    def top_k_gating(self, x):
        router_logits = x @ self.w_gate

        # 轻微噪声用于打破专家选择对称性，仅在训练阶段加入。
        if self.training and self.router_noise > 0:
            router_logits = router_logits + (
                self.router_noise * torch.randn_like(router_logits)
            )

        router_probs = torch.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(
            router_probs,
            k=self.k,
            dim=-1
        )
        top_k_gates = top_k_probs / (
            top_k_probs.sum(dim=-1, keepdim=True) + 1e-8
        )

        gates = torch.zeros_like(router_probs)
        gates = gates.scatter(1, top_k_indices, top_k_gates)

        # importance可导；均匀路由时该损失接近0。
        importance = router_probs.mean(dim=0)
        balance_loss = (
            self.num_experts * torch.sum(importance.pow(2)) - 1.0
        )

        # hard load用于诊断，不参与梯度。
        hard_load = (gates > 0).float().mean(dim=0)
        selection_share = hard_load / (hard_load.sum() + 1e-8)
        router_entropy = -(
            router_probs * torch.log(router_probs + 1e-8)
        ).sum(dim=-1).mean()
        normalized_entropy = router_entropy / math.log(self.num_experts)

        self.last_importance = importance.detach()
        self.last_load = hard_load.detach()
        self.last_selection_share = selection_share.detach()
        self.last_router_entropy = normalized_entropy.detach()

        return gates, balance_loss

    def forward(self, x):
        batch_size, num_patches, feature_size = x.shape
        x_flat = x.reshape(batch_size * num_patches, feature_size)

        gates, balance_loss = self.top_k_gating(x_flat)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = [
            self.experts[i](expert_inputs[i])
            for i in range(self.num_experts)
        ]

        y = x_flat + dispatcher.combine(expert_outputs)
        y = self.rmsnorm(
            y.view(batch_size, num_patches, feature_size)
        )

        return self.act(y), balance_loss


# ---------------------------------------------------------------------------
# Shapelet 嵌入层（支持多变量）
# ---------------------------------------------------------------------------
class ShapeEmbedLayer(nn.Module):
    def __init__(self, seq_len, shape_size, in_chans, embed_dim, stride=4):
        super().__init__()
        self.stride = stride
        num_patches = int((seq_len - shape_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=shape_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


# ---------------------------------------------------------------------------
# Soft Shape 稀疏化辅助函数
# ---------------------------------------------------------------------------
def coml_index(input_indx, dim):
    """计算补集索引"""
    full_idx = torch.arange(dim, device=input_indx.device)
    mask = torch.ones(input_indx.size(0), dim, dtype=torch.bool, device=input_indx.device)

    for i in range(input_indx.size(0)):
        mask[i, input_indx[i]] = False

    complement_idx = torch.stack([full_idx[mask[i]] for i in range(input_indx.size(0))])
    return complement_idx


# ---------------------------------------------------------------------------
# Inception 模块（多尺度特征提取）
# ---------------------------------------------------------------------------
class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=None, bottleneck=True):
        super().__init__()
        if ks is None:
            ks = [1, 3, 5]
        else:
            if isinstance(ks, int):
                ks = [max(1, ks // (2 ** i)) for i in range(3)]
            ks = [max(1, k if k % 2 != 0 else k - 1) for k in ks]

        self.ks = ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else nn.Identity()

        self.convs = nn.ModuleList()
        for k in ks:
            padding = k // 2
            self.convs.append(nn.Conv1d(nf if bottleneck else ni, nf, k,
                                        bias=False, padding=padding))

        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, 1, bias=False)
        )

        total_out_channels = nf * (len(ks) + 1)
        self.bn = nn.BatchNorm1d(total_out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)

        conv_outputs = []
        for i, conv in enumerate(self.convs):
            kernel_size = self.ks[i]
            if x.shape[2] < kernel_size:
                conv_fallback = nn.Conv1d(x.shape[1], conv.out_channels, 1, bias=False).to(x.device)
                conv_outputs.append(conv_fallback(x))
            else:
                conv_outputs.append(conv(x))

        if input_tensor.shape[2] >= 3:
            pool_output = self.maxconvpool(input_tensor)
        else:
            pool_fallback = nn.Conv1d(input_tensor.shape[1], self.maxconvpool[-1].out_channels, 1,
                                      bias=False).to(input_tensor.device)
            pool_output = pool_fallback(input_tensor)

        x = torch.cat(conv_outputs + [pool_output], dim=1)
        return self.act(self.bn(x))


# ---------------------------------------------------------------------------
# SoftShape Learning Block（含 MoE 与 Inception）
# ---------------------------------------------------------------------------
class SoftShapeNet_layer(nn.Module):
    def __init__(self, dim, moe_nets=None, atten_head=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention_head = atten_head
        self.out_drop = nn.Dropout(p=0.15)
        self.moe = moe_nets
        self.inception = InceptionModule(dim, 32, ks=[1, 3, 5])
        self.act = nn.GELU()

    def forward(self, x, end_depth=False, remain_ratio=1.0):
        if remain_ratio < 1.0:
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)

            left_patch_tokens = math.ceil(remain_ratio * x.shape[1])
            left_patch_tokens = min(left_patch_tokens, x.shape[1])

            if left_patch_tokens > 0:
                _, left_idx = torch.topk(attn_x_score, left_patch_tokens, dim=1, largest=True, sorted=True)

                compl_left_indx = coml_index(input_indx=left_idx.squeeze(-1), dim=x.shape[1])
                compl_left_indx = compl_left_indx.unsqueeze(2)

                sorted_left_idx, _ = torch.sort(left_idx, dim=1)
                left_index = sorted_left_idx.expand(-1, -1, x.shape[2])
                compl = compl_left_indx.to(left_index.device)

                non_topk = torch.gather(x * attn_x_score, dim=1, index=compl.expand(-1, -1, x.shape[2]))
                extra_token = torch.sum(non_topk, dim=1, keepdim=True)
                left_x = torch.gather(x * attn_x_score, dim=1, index=left_index)

                x = torch.cat([left_x, extra_token], dim=1)
            else:
                x = torch.mean(x * attn_x_score, dim=1, keepdim=True)

            if x.shape[1] > 0:
                try:
                    incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
                    reshape_incep_x = incep_x.permute(0, 2, 1)
                except Exception as e:
                    fallback_conv = nn.Conv1d(x.shape[2], 32, 1).to(x.device)
                    incep_x = fallback_conv(self.norm2(x).permute(0, 2, 1))
                    reshape_incep_x = incep_x.permute(0, 2, 1)

                temp_moe_x, moe_loss = self.moe(self.norm2(x))
                x = x + temp_moe_x + reshape_incep_x
            else:
                moe_loss = torch.tensor(0.0, device=x.device)
                x = torch.zeros_like(x)
        else:
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)
            x = x * attn_x_score

            try:
                incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
                reshape_incep_x = incep_x.permute(0, 2, 1)
            except Exception as e:
                fallback_conv = nn.Conv1d(x.shape[2], 32, 1).to(x.device)
                incep_x = fallback_conv(self.norm2(x).permute(0, 2, 1))
                reshape_incep_x = incep_x.permute(0, 2, 1)

            moe_loss = 0.0
            x = x + reshape_incep_x

        end_attn_x_score = None
        if end_depth:
            x = self.out_drop(x)
            end_attn_x_score = self.attention_head(x)

        return self.act(x), moe_loss, end_attn_x_score


# ---------------------------------------------------------------------------
# Shapelet Filter（基于 ShapeFormer 的 shapelet 相似度特征）
# ---------------------------------------------------------------------------
class ShapeletFilter(nn.Module):
    def __init__(self, seq_len, num_vars, shapelet_dim=32, num_shapelets=10):
        super().__init__()
        self.num_shapelets = num_shapelets
        self.shapelet_dim = shapelet_dim

        self.shapelets = nn.Parameter(torch.randn(num_shapelets, shapelet_dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_shapelets))

        self.shapelet_proj = nn.Linear(shapelet_dim, shapelet_dim)

        self.input_proj = nn.Sequential(
            nn.Conv1d(num_vars, shapelet_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(shapelet_dim),
            nn.GELU()
        )

    def forward(self, x):
        """x: [B, C, T]"""
        B, C, T = x.shape

        x_proj = self.input_proj(x)
        x_proj = x_proj.permute(0, 2, 1)

        shapelets_proj = self.shapelet_proj(self.shapelets)
        shapelets_expanded = shapelets_proj.unsqueeze(0)

        similarity = torch.einsum('btd,nsd->bts', x_proj, shapelets_expanded)
        similarity = similarity + self.pos_embedding.unsqueeze(0).unsqueeze(0)

        features, _ = torch.max(similarity, dim=1)
        return features.unsqueeze(-1)


# ---------------------------------------------------------------------------
# 主模型：三路径混合架构（SoftShape MoE + 类别特定Transformer + 通用Transformer）
# ---------------------------------------------------------------------------
class SoftShapeTransformerNet(nn.Module):
    def __init__(self, seq_len, shape_size, num_channels, emb_dim, sparse_rate, depth,
                 num_experts, num_classes, stride, transformer_nhead=4,
                 transformer_layers=2, moe_top_k=2, sparse_ramp_epoch=20):
        super().__init__()

        self.seq_len = seq_len
        self.shape_size = shape_size
        self.num_channels = num_channels
        self.emb_dim = emb_dim
        self.sparse_rate = sparse_rate
        self.depth = depth
        self.num_experts = int(num_experts)
        self.moe_top_k = int(moe_top_k)
        self.sparse_ramp_epoch = max(1, int(sparse_ramp_epoch))
        self.current_sparse_progress = 0.0
        self.current_sparse_rate = 0.0
        self.num_classes = num_classes
        self.stride = stride

        # ========== 路径1: SoftShape MoE 路径 ==========
        self.shape_embed = ShapeEmbedLayer(
            seq_len=self.seq_len, shape_size=self.shape_size,
            in_chans=self.num_channels, embed_dim=self.emb_dim, stride=stride
        )

        if self.shape_embed.num_patches <= 0:
            raise ValueError(
                f"num_patches必须大于0，当前为{self.shape_embed.num_patches}。请检查seq_len={seq_len}, shape_size={shape_size}, stride={stride}")

        self.attention_head = nn.Sequential(
            nn.Linear(self.emb_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.moe = MoE_Block(
            input_size=self.emb_dim,
            output_size=self.emb_dim,
            num_experts=self.num_experts,
            hidden_size=self.emb_dim,
            k=self.moe_top_k
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.shape_embed.num_patches, self.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.15)

        self.sparse_ratio_d = [x.item() for x in torch.linspace(0, self.sparse_rate, self.depth)]

        self.shape_blocks = nn.ModuleList([
            SoftShapeNet_layer(dim=self.emb_dim, moe_nets=self.moe, atten_head=self.attention_head)
            for _ in range(self.depth)]
        )

        # ========== 路径2: 类别特定 Transformer 路径 ==========
        self.shapelet_filter = ShapeletFilter(
            seq_len=seq_len,
            num_vars=num_channels,
            shapelet_dim=emb_dim // 2,
            num_shapelets=min(10, num_experts)
        )

        self.class_specific_transformer = TransformerEncoder(
            embed_dim=emb_dim // 2,
            num_heads=transformer_nhead,
            num_layers=min(2, transformer_layers)
        )

        # ========== 路径3: 通用 Transformer 路径 ==========
        self.generic_conv = nn.Sequential(
            nn.Conv1d(num_channels, emb_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(emb_dim // 2),
            nn.GELU(),
            nn.Conv1d(emb_dim // 2, emb_dim // 2, kernel_size=1, padding=0),
            nn.BatchNorm1d(emb_dim // 2),
            nn.GELU()
        )

        self.generic_transformer = TransformerEncoder(
            embed_dim=emb_dim // 2,
            num_heads=transformer_nhead,
            num_layers=min(2, transformer_layers)
        )

        # ========== 特征融合与分类头 ==========
        total_feat_dim = emb_dim * 2
        self.fusion_proj = nn.Linear(total_feat_dim, emb_dim)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, num_epoch_i=100, warm_up_epoch=50):
        B, C, T = x.shape
        total_moe_loss = 0.0

        # ========== 路径1: SoftShape MoE 路径 ==========
        try:
            x1 = self.shape_embed(x)
            x1 = x1 + self.pos_embed
            x1 = self.pos_drop(x1)
        except Exception as e:
            print(f"Error in path 1 (shape embed): {e}")
            x1 = x.mean(dim=1, keepdim=True).repeat(1, 5, 1)

        moe_loss = None
        d = 0
        end_attn_x_score = None

        # 前warm_up_epoch轮不稀疏；随后在sparse_ramp_epoch轮内从0线性增加到目标稀疏率。
        if num_epoch_i < warm_up_epoch:
            sparse_progress = 0.0
        else:
            sparse_progress = min(
                1.0,
                (num_epoch_i - warm_up_epoch + 1)
                / float(self.sparse_ramp_epoch)
            )

        self.current_sparse_progress = sparse_progress
        self.current_sparse_rate = self.sparse_rate * sparse_progress

        for shape_blk in self.shape_blocks:
            layer_sparse_rate = self.sparse_ratio_d[d] * sparse_progress
            depth_remain_ratio = 1.0 - layer_sparse_rate

            judge_end = False
            if (d + 1) == self.depth:
                judge_end = True

            try:
                x1, _temp_mloss, end_attn_x_score = shape_blk(
                    x1, end_depth=judge_end, remain_ratio=depth_remain_ratio
                )
                d = d + 1

                if moe_loss is None:
                    moe_loss = _temp_mloss
                else:
                    moe_loss = moe_loss + _temp_mloss
            except Exception as e:
                print(f"Error in shape block {d}: {e}")
                d += 1
                if moe_loss is None:
                    moe_loss = torch.tensor(0.0, device=x.device)
                continue

        if end_attn_x_score is not None:
            x1_weighted = x1 * end_attn_x_score
        else:
            x1_weighted = x1

        if x1_weighted.shape[1] > 0:
            path1_feat = torch.mean(x1_weighted, dim=1)
        else:
            path1_feat = torch.zeros(B, self.emb_dim, device=x.device)

        # ========== 路径2: 类别特定 Transformer 路径 ==========
        try:
            shapelet_features = self.shapelet_filter(x)
            shapelet_features = shapelet_features.squeeze(-1)

            shapelet_pos = torch.arange(shapelet_features.shape[1], device=x.device).unsqueeze(0).repeat(B, 1)
            pos_embed = torch.zeros(shapelet_features.shape[1], self.emb_dim // 2, device=x.device)
            shapelet_features_embedded = shapelet_features.unsqueeze(-1)
            shapelet_features_expanded = shapelet_features_embedded.expand(-1, -1,
                                                                           self.emb_dim // 2)

            shapelet_features_expanded = shapelet_features_expanded + pos_embed.unsqueeze(0)

            path2_feat = self.class_specific_transformer(shapelet_features_expanded)
            path2_feat = path2_feat.mean(dim=1)
        except Exception as e:
            print(f"Error in path 2 (shapelet filter): {e}")
            path2_feat = torch.zeros(B, self.emb_dim // 2, device=x.device)

        # ========== 路径3: 通用 Transformer 路径 ==========
        try:
            conv_features = self.generic_conv(x)
            conv_features = conv_features.permute(0, 2, 1)

            seq_pos = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
            pos_embed = torch.zeros(T, self.emb_dim // 2, device=x.device)
            conv_features = conv_features + pos_embed.unsqueeze(0)

            path3_feat = self.generic_transformer(conv_features)
            path3_feat = path3_feat.mean(dim=1)
        except Exception as e:
            print(f"Error in path 3 (generic transformer): {e}")
            path3_feat = torch.zeros(B, self.emb_dim // 2, device=x.device)

        # ========== 特征融合与分类 ==========
        combined_feat = torch.cat([path1_feat, path2_feat, path3_feat], dim=1)
        fused_feat = self.fusion_proj(combined_feat)
        logits = self.classifier(fused_feat)

        if moe_loss is None:
            moe_loss = torch.tensor(0.0, device=x.device)

        return logits, moe_loss


# ---------------------------------------------------------------------------
# 地层序列 CRF 约束模型（整井单向层序软约束）
# ---------------------------------------------------------------------------
class StratigraphyCRFNet(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        transition_penalty=-3.0
    ):
        super().__init__()

        self.base_model = base_model
        self.crf = CRF(
            num_classes,
            batch_first=True
        )

        self.transition_penalty = float(
            transition_penalty
        )

        self.register_buffer(
            "invalid_transition_mask",
            self._build_stratigraphy_mask()
        )

    def _build_stratigraphy_mask(self):
        """
        采用整井CRF的单向层序软约束。

        标签编码：嘉一=0、嘉三=1、嘉二=2、嘉五=3、嘉四=4。
        slice编号递增方向：嘉五(3)->嘉四(4)->嘉三(1)->嘉二(2)->嘉一(0)。
        """
        mask = torch.ones(
            (self.crf.num_tags, self.crf.num_tags),
            dtype=torch.bool
        )

        valid_transitions = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
            (3, 4),  # 嘉五 -> 嘉四
            (4, 1),  # 嘉四 -> 嘉三
            (1, 2),  # 嘉三 -> 嘉二
            (2, 0),  # 嘉二 -> 嘉一
        ]

        for start_tag, end_tag in valid_transitions:
            mask[start_tag, end_tag] = False

        return mask

    def _apply_transition_constraints(self):
        """对非单向层序转移施加固定软惩罚。"""
        with torch.no_grad():
            self.crf.transitions.data.masked_fill_(
                self.invalid_transition_mask,
                self.transition_penalty
            )

    def get_emissions(
        self,
        x_seq,
        num_epoch_i=100,
        warm_up_epoch=50
    ):
        """
        计算每个切片的发射分数。

        x_seq: [B, S, C, T]

        返回
        ----
        emissions: [B, S, num_classes]
        moe_loss: scalar tensor
        """
        B, S, C, T = x_seq.shape

        x_flat = x_seq.reshape(B * S, C, T)

        logits_flat, moe_loss = self.base_model(
            x_flat,
            num_epoch_i=num_epoch_i,
            warm_up_epoch=warm_up_epoch
        )

        emissions = logits_flat.reshape(B, S, -1)

        if not torch.is_tensor(moe_loss):
            moe_loss = emissions.new_tensor(float(moe_loss))

        return emissions, moe_loss

    def decode(self, emissions):
        """使用同一组 emissions 进行 Viterbi 解码。"""
        self._apply_transition_constraints()

        best_paths = self.crf.decode(emissions)

        return torch.as_tensor(
            best_paths,
            dtype=torch.long,
            device=emissions.device
        )

    def forward(
        self,
        x_seq,
        labels=None,
        num_epoch_i=100,
        warm_up_epoch=50,
        return_emissions=False
    ):
        """
        x_seq:  [B, S, C, T]
        labels: [B, S]

        传入 labels 时返回 CRF loss、MoE loss 和可选 emissions；
        未传入 labels 时返回 Viterbi 解码结果。
        """
        emissions, moe_loss = self.get_emissions(
            x_seq=x_seq,
            num_epoch_i=num_epoch_i,
            warm_up_epoch=warm_up_epoch
        )

        self._apply_transition_constraints()

        if labels is not None:
            crf_loss = -self.crf(
                emissions,
                labels,
                reduction="token_mean"
            )

            if return_emissions:
                return crf_loss, moe_loss, emissions

            return crf_loss, moe_loss

        best_paths = self.decode(emissions)

        if return_emissions:
            return best_paths, moe_loss, emissions

        return best_paths, moe_loss

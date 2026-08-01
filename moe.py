"""
MoE FFN —— DeepSeek-V3 风格的现代混合专家实现

相比 dense FFN 的核心变化：
  dense:  每个 token 都过同一个大 FFN
  MoE:    每个 token 只激活 N 个专家中的 K 个（稀疏激活）

四个关键设计（都是 DeepSeek-V2/V3 的做法，现已成为业界主流）：
  1. 细粒度专家：专家做小做多（256 个小专家，而不是 8 个大专家）
  2. 共享专家：1~2 个恒定激活的专家，承载通用知识
  3. Sigmoid 路由：V3 把 softmax 换成 sigmoid，专家数很多时更稳定
  4. 无辅助损失负载均衡：用可学习 bias 调路由，不再靠 aux loss 干扰主目标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class MoEConfig:
    dim: int = 512                  # 模型隐层维度
    n_routed_experts: int = 64      # 路由专家总数（V3 是 256）
    n_shared_experts: int = 1       # 共享专家数（恒定激活）
    n_activated: int = 6            # 每个 token 激活几个路由专家（V3 是 8）
    moe_inter_dim: int = 352        # 单个路由专家的中间维度（细粒度：远小于 dense 的 dim_ff）
    shared_inter_dim: int = 704     # 共享专家的中间维度（通常比路由专家大）

    # 分组限制路由（限制跨节点通信，V3 的 node-limited routing）
    n_groups: int = 8               # 把路由专家分成几组（对应物理节点）
    topk_groups: int = 3            # 每个 token 最多命中几个组

    route_scale: float = 2.5        # 路由权重的整体缩放（V3 用 2.5）
    bias_update_speed: float = 1e-3 # 负载均衡 bias 的更新步长 γ


# ─────────────────────────────────────────────────────────────
# 1. 单个专家（就是一个小号 SwiGLU FFN）
# ─────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """
    结构和 dense FFN 完全一样，只是 inter_dim 小得多。

    细粒度专家的意义：
      粗粒度  8 个专家 × inter_dim 2048，选 2 个  → 组合数 C(8,2)=28
      细粒度 64 个专家 × inter_dim  352，选 6 个  → 组合数 C(64,6)≈7000万
    激活参数量相近，但专家组合的表达能力天差地别。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.W_GATE = nn.Linear(dim, inter_dim, bias=False)
        self.W_UP   = nn.Linear(dim, inter_dim, bias=False)
        self.W_DOWN = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        return self.W_DOWN(F.silu(self.W_GATE(x)) * self.W_UP(x))


# ─────────────────────────────────────────────────────────────
# 2. 路由器（Router / Gate）
# ─────────────────────────────────────────────────────────────

class Router(nn.Module):
    """
    决定每个 token 该送给哪几个专家。

    输入: (N, dim) 的 token 表示
    输出: topk_idx    (N, K) 选中的专家编号
          topk_weight (N, K) 对应的加权系数
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        # 路由权重矩阵：每个专家一个 dim 维的"质心"向量
        # token 和哪个质心越接近，就越该送给那个专家
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # ★ 无辅助损失负载均衡的核心：每个专家一个 bias
        # 注意是 buffer 不是 Parameter —— 它不参与反向传播，
        # 而是在每个训练 step 后根据实际负载手动调整（见 update_bias）
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))
        # 统计本 step 每个专家收到多少 token，用于更新 bias
        self.register_buffer("load_counter", torch.zeros(cfg.n_routed_experts))

    def forward(self, x):
        cfg = self.cfg
        N = x.shape[0]

        # ① 打分：token 和每个专家质心做内积
        logits = F.linear(x.float(), self.weight.float())      # (N, E)

        # ② Sigmoid 而非 Softmax
        #    softmax 在专家数达到 256 时，分数会被摊得极薄且互相耦合；
        #    sigmoid 让每个专家独立打分，数值更稳定
        scores = logits.sigmoid()                              # (N, E)

        # ③ 选 topk 时用 scores + bias，但最终权重用原始 scores
        #    bias 只影响"选谁"，不影响"给多大权重"——这是无损失均衡的关键
        scores_for_choice = scores + self.expert_bias          # (N, E)

        # ④ 分组限制路由：先选组，再在选中的组内选专家
        if cfg.n_groups > 1:
            grouped = scores_for_choice.view(N, cfg.n_groups, -1)  # (N, G, E/G)
            # 每组的代表分 = 组内 top2 之和（V3 的做法，比取 max 更鲁棒）
            group_score = grouped.topk(2, dim=-1)[0].sum(dim=-1)   # (N, G)
            # 选出得分最高的 topk_groups 个组
            group_idx = group_score.topk(cfg.topk_groups, dim=-1)[1]  # (N, tg)
            # 未选中的组整体屏蔽掉
            group_mask = torch.zeros_like(group_score).scatter_(1, group_idx, 1.0)
            expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape(N, -1)
            scores_for_choice = scores_for_choice.masked_fill(expert_mask == 0, float("-inf"))

        # ⑤ 在允许的范围内选 top-K 个专家
        topk_idx = scores_for_choice.topk(cfg.n_activated, dim=-1)[1]   # (N, K)

        # ⑥ 取权重时回到"干净的" scores（不含 bias）
        topk_weight = scores.gather(1, topk_idx)                        # (N, K)
        # 归一化，让 K 个权重和为 1，再整体放大 route_scale
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        topk_weight = topk_weight * cfg.route_scale

        # ⑦ 训练时累计负载统计，供 update_bias 使用
        if self.training:
            with torch.no_grad():
                self.load_counter += torch.bincount(
                    topk_idx.flatten(), minlength=cfg.n_routed_experts
                ).float()

        return topk_idx, topk_weight.type_as(x)

    @torch.no_grad()
    def update_bias(self):
        """
        无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing）。

        在每个 optimizer.step() 之后调用一次：
          - 收到 token 多于平均值的专家 → bias 调低 → 下次少被选中
          - 收到 token 少于平均值的专家 → bias 调高 → 下次多被选中

        为什么优于传统 aux loss：
          aux loss 会往主损失里掺一项和"学好语言"无关的目标，
          损害模型质量；bias 调整完全在梯度之外，不污染主目标。
        """
        if self.load_counter.sum() == 0:
            return
        mean_load = self.load_counter.mean()
        # 超载为正、欠载为负，取符号 → bias 反向移动固定步长
        overload = torch.sign(self.load_counter - mean_load)
        self.expert_bias -= self.cfg.bias_update_speed * overload
        self.load_counter.zero_()

    @torch.no_grad()
    def load_balance_stat(self):
        """返回变异系数 CV，用于监控均衡程度。理想值接近 0，>1 说明严重倾斜。"""
        if self.load_counter.sum() == 0:
            return 0.0
        return (self.load_counter.std() / self.load_counter.mean().clamp(min=1e-9)).item()


# ─────────────────────────────────────────────────────────────
# 3. MoE FFN（替换原来的 dense FFN）
# ─────────────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """
    输出 = 共享专家(x)  +  Σ_{k∈topK} weight_k · 专家_k(x)

    共享专家为什么必要：
      没有它时，"标点怎么用""基本语法"这类所有 token 都需要的通用知识，
      会被迫在每个路由专家里各存一份 → 参数冗余。
      把通用知识隔离到恒定激活的共享专家后，路由专家才能真正专业化。
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        self.router = Router(cfg)

        # 路由专家：稀疏激活，每个 token 只走其中 K 个
        self.experts = nn.ModuleList([
            Expert(cfg.dim, cfg.moe_inter_dim) for _ in range(cfg.n_routed_experts)
        ])

        # 共享专家：所有 token 都走，等价于一个小号 dense FFN
        self.shared_expert = Expert(cfg.dim, cfg.shared_inter_dim * cfg.n_shared_experts)

    def forward(self, x):
        # x: (B, T, dim) → 拍平成 token 列表，路由是逐 token 的，不关心序列结构
        shape = x.shape
        x = x.view(-1, shape[-1])                          # (N, dim), N = B*T

        topk_idx, topk_weight = self.router(x)             # (N, K), (N, K)

        # 稀疏部分
        y = self._dispatch(x, topk_idx, topk_weight)       # (N, dim)
        # 加上共享专家（恒定激活）
        y = y + self.shared_expert(x)

        return y.view(shape)

    def _dispatch(self, x, topk_idx, topk_weight):
        """
        把 token 分发给各自的专家，算完再按权重聚合回来。

        朴素写法是 for 每个 token 找它的专家，会极慢。
        这里用"按专家排序后分段处理"：让同一个专家的所有 token 凑成一批，
        一次矩阵乘算完 —— 这是所有 MoE 实现的通用套路。
        """
        N, K = topk_idx.shape
        flat_idx = topk_idx.flatten()                      # (N*K,) 每条 = 一次"token→专家"分配

        # 按专家编号排序，让同专家的分配聚在一起
        order = flat_idx.argsort()                         # (N*K,)
        # 每个专家收到多少条分配 → 累加得到每段的结束位置
        seg_end = torch.bincount(flat_idx, minlength=self.cfg.n_routed_experts).cumsum(0)
        # order 里的下标 // K 就是原 token 的行号
        token_of = order // K

        out = torch.zeros_like(x)
        start = 0
        for eid, end in enumerate(seg_end.tolist()):
            if start == end:                               # 这个专家这批没分到 token
                continue
            rows = token_of[start:end]                     # 该专家要处理的 token 行号
            expert_out = self.experts[eid](x[rows])        # 批量算，(n_e, dim)
            # 乘上各自的路由权重
            expert_out = expert_out * topk_weight.flatten()[order[start:end]].unsqueeze(-1)
            # 累加回原位置（同一 token 被多个专家选中时会累加多次，这正是我们要的）
            out.index_add_(0, rows, expert_out.type_as(out))
            start = end

        return out


# ─────────────────────────────────────────────────────────────
# 4. 参数量对比工具
# ─────────────────────────────────────────────────────────────

def compare_params(cfg: MoEConfig, dense_inter_dim: int = 2048):
    """对比 dense FFN 和 MoE FFN 的总参数量 / 激活参数量"""
    dense_total = 3 * cfg.dim * dense_inter_dim

    routed_one = 3 * cfg.dim * cfg.moe_inter_dim
    routed_total = routed_one * cfg.n_routed_experts
    shared_total = 3 * cfg.dim * cfg.shared_inter_dim * cfg.n_shared_experts
    router_total = cfg.dim * cfg.n_routed_experts

    moe_total = routed_total + shared_total + router_total
    moe_active = routed_one * cfg.n_activated + shared_total + router_total

    print(f"{'':>22}{'总参数':>14}{'激活参数':>14}")
    print(f"{'dense FFN':>22}{dense_total/1e6:>12.2f}M{dense_total/1e6:>13.2f}M")
    print(f"{'MoE FFN':>22}{moe_total/1e6:>12.2f}M{moe_active/1e6:>13.2f}M")
    print(f"{'倍率':>22}{moe_total/dense_total:>12.2f}x{moe_active/dense_total:>13.2f}x")


# ─────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MoEConfig()
    moe = MoEFFN(cfg)

    x = torch.randn(2, 128, cfg.dim)
    y = moe(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    assert y.shape == x.shape

    # 反向传播能否跑通
    y.sum().backward()
    print("backward: ok")
    print()

    compare_params(cfg)
    print()

    # 模拟几步训练，观察负载均衡 bias 是否在起作用
    moe.train()
    print("负载均衡演化（CV 越小越均衡）：")
    for step in range(5):
        moe(torch.randn(4, 256, cfg.dim))
        cv = moe.router.load_balance_stat()
        moe.router.update_bias()
        print(f"  step {step}: CV = {cv:.4f}, "
              f"bias 范围 [{moe.router.expert_bias.min():+.4f}, "
              f"{moe.router.expert_bias.max():+.4f}]")

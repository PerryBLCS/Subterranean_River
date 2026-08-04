"""
GAD-TCN (Gated Attention Dilated TCN)
门控注意力膨胀时序卷积网络
商业银行信用贷款风险预警模型

===========================================================================
三大创新模块：
  HDG  — 层级膨胀门控：按膨胀率、层深、激活饱和度衡量每层贡献
  CCMA — 时序-静态交叉融合：低秩 U*V^T 关联 + 多头注意力
  DAFL — 膨胀感知动态损失：分支熵正则 + 动态 gamma

七指标平等评估：
  AUC-ROC / AUC-PR / Brier Score / Accuracy / F1 / Sensitivity / Specificity
  消融实验中每个模块的贡献通过全部七个指标的delta综合判断

调用入口：
  python essay_tcn.py                # 进入交互选择
  python essay_tcn.py --dataset taiwan --epochs 200
  python essay_tcn.py --dataset home --epochs 100

Python API:
  from essay_tcn import run_experiment
  model, metrics, history = run_experiment(dataset_name='taiwan')

数据统一通过 URL 加载，不提供本地路径参数。
===========================================================================
"""

# ============================================================
# 一、加载 —— 依赖与全局设置
# ============================================================
import numpy as np
import pandas as pd
import os, sys, copy, argparse, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, fbeta_score, recall_score, precision_score, confusion_matrix,
    balanced_accuracy_score, brier_score_loss, log_loss
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

TAIWAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
HOME_CREDIT_URL = (
    "https://www.kaggle.com/c/home-credit-default-risk/data"
)
DATA_CACHE_DIR = 'data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def download_and_extract(url, cache_dir=DATA_CACHE_DIR):
    """从 URL 下载数据文件并解压（如为 zip），返回本地可用路径。"""
    pass


def ask_yes_no(question, default=True):
    """交互式 Y/N 提问，空输入使用默认值。"""
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = input(question + suffix).strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("请输入 y 或 n。")


def ask_gate_style(default="full"):
    """交互式选择 CCMA 门控样式；compare 表示三种都跑并报告最优。"""
    prompt = (
        "\n选择 CCMA 融合门控样式:\n"
        "  score   - 3参数关联分数门控（最稳）\n"
        "  mlp     - 2维小MLP门控（折中）\n"
        "  full    - 385维完整MLP门控（台湾实测更优，默认）\n"
        "  compare - 三种都跑，最后报告最优\n"
        f"请输入 score/mlp/full/compare [{default}]: "
    )
    while True:
        answer = input(prompt).strip().lower()
        if not answer:
            return default
        if answer in ("score", "mlp", "full", "compare"):
            return answer
        print("请输入 score、mlp、full 或 compare。")


def interactive_experiment_setup(default_dataset=None):
    """启动交互：选择数据集、实验开关、CCMA门控样式，返回参数字典。"""
    if default_dataset is None:
        default_dataset = input("选择数据集 (taiwan/home/both) [taiwan]: ").strip().lower() or "taiwan"
    if default_dataset not in ("taiwan", "home", "both"):
        raise ValueError("dataset 必须是 taiwan、home 或 both。")

    return {
        "dataset_name": default_dataset,
        "run_baselines": ask_yes_no("是否运行基线对比 (ML+DL)？", default=True),
        "run_ablation": ask_yes_no("是否运行消融实验？", default=True),
        "run_imbalance": ask_yes_no("是否运行类别不平衡鲁棒性？", default=True),
        "run_cross_dataset": ask_yes_no("是否运行跨数据集泛化？", default=True),
        "run_history_length": ask_yes_no("是否运行历史长度敏感性？", default=True),
        "run_rank_sensitivity": ask_yes_no("是否运行 CCMA rank 敏感性？", default=True),
        "run_shap": ask_yes_no("是否运行 SHAP 可解释性？", default=True),
        "make_plots": ask_yes_no("是否生成 PNG 图表？", default=True),
        "save_results": ask_yes_no("是否保存结果？", default=True),
        "gate_style": ask_gate_style(),
        "gate_hidden": 8,
    }


def set_seed(seed, deterministic=True):
    """Set random seed for reproducibility across Python, NumPy, PyTorch, and cuDNN."""
    pass

def specificity_score(y_true, y_pred):
    """计算特异性 TN/(TN+FP)。"""
    pass

def scalar_metrics(metrics):
    """过滤ndarray，保留标量用于JSON序列化。"""
    pass


def compute_metrics(y_true, probabilities, predictions=None, threshold=0.5):
    """统一计算七项评估指标，返回标量与原始数组。"""
    pass


def split_and_scale(static_features, temporal_features, y,
                    scaler, temporal_scaler,
                    static_encoder=None,
                    temporal_categorical_indices=(0,),
                    test_size=0.15, val_size=0.10, calibration_size=0.10,
                    split_seed=42):
    """四层切分、类别编码与标准化，返回统一结果字典。"""
    pass


# ============================================================
# 二、数据集处理 —— Taiwan Credit + Home Credit
# ============================================================

class CreditDataLoader:
    """
    Taiwan Credit数据集加载器（30,000样本，~22%违约率）。

    数据来源: UCI Machine Learning Repository
      URL: TAIWAN_CREDIT_URL

    时序5通道×6月（最早月→最近月）:
      payment_status, signed_log_bill_amount, signed_log_payment_amount,
      bill_to_limit, payment_to_bill

    静态: LIMIT_BAL + AGE + one-hot(SEX/EDUCATION/MARRIAGE) + 9个工程特征
    金额处理: signed_log1p = sign(x)*log(1+|x|)

    调用:
      loader = CreditDataLoader()
      s,t,y = loader.load_taiwan_credit("taiwan_credit.xls")
      (Xstr,Xttr,ytr,Xsv,Xtv,yv,Xste,Xtte,yte) = loader.preprocess(s,t,y)
    """
    def __init__(self): pass
    def load_taiwan_credit(self, url=None):
        """从 URL 加载，返回 static(N,S), temporal(N,6,5), y(N,) 1=违约"""
        pass
    def _encode_static_from_training(self, train_idx, *evaluation_indices):
        """训练集拟合one-hot词表，评估集对齐。内部方法。"""
        pass
    @staticmethod
    def _scaler_state(scaler):
        """序列化StandardScaler参数，用于部署导出。"""
        pass
    def export_preprocessing_state(self):
        """导出部署所需参数：特征名、缩放器参数、类别映射。"""
        pass
    def preprocess(self, static_features, temporal_features, y,
                   test_size=0.15, val_size=0.10, calibration_size=0.10,
                   split_seed=42):
        """四层分层分割+标准化+one-hot编码。
        校准集存储在self.calibration_data_。
        返回: (Xst_tr,Xt_tr,yt_tr, Xst_v,Xt_v,yt_v, Xst_te,Xt_te,yt_te)"""
        pass


class CreditDataset(Dataset):
    """PyTorch Dataset。batch={'static':[S],'temporal':[T,F],'label':scalar}
    调用: ds=CreditDataset(Xst,Xt,y); batch=ds[0]"""
    def __init__(self, static_features, temporal_features, labels): pass
    def __len__(self): pass
    def __getitem__(self, idx): pass


class HomeCreditLoader:
    """
    Home Credit数据集加载器（307K申请，7表全面特征工程）。

    数据来源: Kaggle Home Credit Default Risk
      URL: HOME_CREDIT_URL（自动尝试下载/解压）

    7表: application_train/test, bureau+bureau_balance, credit_card_balance,
         installments_payments, POS_CASH_balance, previous_application

    时序: 默认最近24个月，12通道 = 6(信用卡) + 3(征信) + 3(还款行为)

    调用:
      loader = HomeCreditLoader(url=None, max_temporal_steps=24)
      s,t,y = loader.build_all_features()
      (Xstr,Xttr,ytr,Xsv,Xtv,yv,Xste,Xtte,yte) = loader.preprocess(s,t,y)
    """
    def __init__(self, url=None, max_temporal_steps=24):
        """url: 下载链接，默认 HOME_CREDIT_URL。max_temporal_steps: 时序最大月数"""
        pass
    def load_application(self):
        """返回 static_base(N,S_base), y(N,)"""
        pass
    def load_credit_card_balance(self):
        """信用卡月度时序。6通道。返回 cc_temporal(N,T,6)"""
        pass
    def load_bureau(self):
        """信用历史时序。3通道。返回 bureau_temporal(N,T,3)"""
        pass
    def load_installments_and_pos(self):
        """还款行为时序。3通道。返回 repayment_temporal(N,T,3)"""
        pass
    def load_previous_application(self):
        """历史申请聚合。返回 prev_static(N,P)"""
        pass
    def build_all_features(self):
        """编排7表加载→拼接。返回 static(N,S), temporal(N,T,12), y(N,)"""
        pass
    def preprocess(self, static_features, temporal_features, y,
                   test_size=0.15, val_size=0.10, calibration_size=0.10,
                   split_seed=42):
        """与CreditDataLoader.preprocess相同接口。四层分割+标准化。"""
        pass


# ============================================================
# 三、基本框架 —— TCN基础架构
# ============================================================

class CausalConv1d(nn.Module):
    """
    因果膨胀卷积（只看过去，不看未来）。
    左侧padding=(k-1)*d确保output[t]仅依赖input[<=t]。
    WeightNorm稳定训练。

    调用: conv=CausalConv1d(128,128,kernel_size=3,dilation=2)
         out=conv(x)  # x:[B,128,T]->out:[B,128,T]
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation))
    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    因果膨胀卷积残差块（TCN基本单元）。

    内部: CausalConv1d(d)×2 -> 残差 -> HDG门控 -> LayerNorm
    感受野: RF=1+2*(k-1)*d   d=1->5, d=2->9, d=4->17, d=8->33

    调用: block=TCNBlock(128,128,dilation=2,depth=1,num_layers=4,use_hdg=True)
         out,hdg_stats=block(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 dropout=0.2, depth=0, num_layers=4, use_hdg=True):
        super().__init__()
        self.dilation = dilation
        self.depth = depth
        self.use_hdg = use_hdg
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        if use_hdg:
            self.hdg = HierarchicalDilatedGating(
                out_channels, dilation=dilation, depth=depth, num_layers=num_layers)
    def forward(self, x):
        """返回 out[B,T,out_C], hdg_stats dict|None"""
        x_t = x.transpose(1, 2)                     # [B,C,T]
        h = self.dropout(self.relu(self.conv1(x_t)))
        h = self.dropout(self.relu(self.conv2(h)))  # [B,C,T] 残差块学到的信号
        shortcut = self.skip(x_t)                   # [B,C,T] 恒等/投影旁路
        if self.use_hdg:
            gated, hdg_stats = self.hdg(
                shortcut.transpose(1, 2), h.transpose(1, 2))
            return self.norm(gated), hdg_stats
        out = (h + shortcut).transpose(1, 2)
        return self.norm(out), None


class TCNEncoder(nn.Module):
    """
    TCN编码器 — 串行堆叠 或 并行多分支。

    parallel(默认): 同输入进d=1,2,4,8四路独立TCNBlock。
      -> List[[B,T,D]]x4 + hdg_stats_list
    stacked: 逐层串行，dilation翻倍。
      -> [B,T,D] + hdg_stats_list

    调用: encoder=TCNEncoder(128,128,4,mode='parallel')
         outputs,hdg_list=encoder(x)
    """
    def __init__(self, input_dim, hidden_dim, num_layers=4, kernel_size=3,
                 base_dilation=1, dropout=0.2, use_hdg=True, mode='parallel',
                 dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [base_dilation * (2 ** i) for i in range(num_layers)]
        self.dilation_rates = list(dilation_rates)
        self.mode = mode
        if mode == 'parallel':
            self.blocks = nn.ModuleList([
                TCNBlock(input_dim, hidden_dim, kernel_size, d, dropout,
                         depth=i, num_layers=len(self.dilation_rates), use_hdg=use_hdg)
                for i, d in enumerate(self.dilation_rates)
            ])
        else:
            self.blocks = nn.ModuleList([
                TCNBlock(input_dim if i == 0 else hidden_dim, hidden_dim,
                         kernel_size, d, dropout, depth=i,
                         num_layers=len(self.dilation_rates), use_hdg=use_hdg)
                for i, d in enumerate(self.dilation_rates)
            ])
    def forward(self, x):
        hdg_list = []
        if self.mode == 'parallel':
            outputs = []
            for block in self.blocks:
                out, stats = block(x)
                outputs.append(out)
                hdg_list.append(stats)
            return outputs, hdg_list
        h = x
        for block in self.blocks:
            h, stats = block(h)
            hdg_list.append(stats)
        return h, hdg_list


# ============================================================
# 四、算法创新 —— HDG + CCMA + DAFL
# ============================================================

# ----- 创新一：层次膨胀门控 (HDG) -----

class HierarchicalDilatedGating(nn.Module):
    """
    层次膨胀门控(HDG) — TCN特有创新。

    核心: 堆叠越深不一定越好。每层根据三个结构信号动态决定残差贡献:
      1. 膨胀率d     — 感受野大小（log2归一化）
      2. 层深depth   — 越深越易过拟合（逐层递减偏置）
      3. 激活饱和度  — 从残差块输出计算：太平=没学到东西

    防过拟合: 逐层递减偏置（深层默认保守，信号强才激活）
      gate = sigmoid(MLP([d_norm, depth_norm, sat_mean, sat_var]))
      output = x + gate * residual     # x=投影后的shortcut, residual=卷积块输出

    位置: TCNBlock残差连接之后、LayerNorm之前。

    调用: hdg=HierarchicalDilatedGating(128,dilation=4,depth=2,num_layers=4)
         gated,stats=hdg(x,residual)
    """
    def __init__(self, hidden_dim, dilation, depth, num_layers=4, bottleneck=None,
                 gate_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.depth = depth
        self.num_layers = num_layers
        d_norm = float(np.log2(max(dilation, 1))) / max(num_layers - 1, 1)
        depth_norm = float(depth) / max(num_layers - 1, 1)
        self.register_buffer('d_norm', torch.tensor(d_norm))
        self.register_buffer('depth_norm', torch.tensor(depth_norm))
        self.gate_net = nn.Sequential(
            nn.Linear(4, bottleneck or 16),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(bottleneck or 16, 1),
        )
        with torch.no_grad():
            # 深层默认更保守：偏置随depth线性变负
            self.gate_net[-1].bias.fill_(-0.5 - 1.0 * depth_norm)
    def _compute_activation_saturation(self, x):
        """返回逐token mean[B,T], var[B,T]"""
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, unbiased=False)
        return mean, var
    def forward(self, x, residual):
        """返回 gated_output[B,T,D], gate_stats dict"""
        sat_mean, sat_var = self._compute_activation_saturation(residual)
        gate_in = torch.stack([
            self.d_norm.expand_as(sat_mean),
            self.depth_norm.expand_as(sat_mean),
            torch.tanh(sat_mean),
            torch.tanh(sat_var / (float(self.hidden_dim) ** 0.5)),
        ], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_in))      # [B,T,1]
        out = x + gate * residual
        gate_stats = {
            'gate_mean': float(gate.mean().item()),
            'gate_std': float(gate.std().item()),
            'gate_min': float(gate.min().item()),
            'gate_max': float(gate.max().item()),
            'sat_mean': float(sat_mean.mean().item()),
            'sat_var': float(sat_var.mean().item()),
            'd_norm': float(self.d_norm.item()),
            'depth_norm': float(self.depth_norm.item()),
        }
        return out, gate_stats


# ----- 创新二：交叉拼接多头注意力 (CCMA) -----

class LowRankCrossFusion(nn.Module):
    """
    低秩交叉关联矩阵 — TCN特有创新（CCMA第一阶段）。

    核心: 不让静态特征盲目展开到每个时间步。用U*V^T(秩=4)强约束交叉空间。

    U:[static_dim,rank=4]  V:[temporal_dim,rank=4]

    三条通路+门控融合（交叉项只做增量，独立通路永远保留）:
      cross   = (static@U) * (temporal@V) -> Linear -> [B,T,D]
      s_path  = Linear(static).expand     -> [B,T,D]
      t_path  = Linear(temporal)          -> [B,T,D]
      cos     = cosine_similarity(static_lr, temporal_lr)      # [-1,1] 方向一致性
      mag     = sum(static_lr * temporal_lr)                   # 共同变化幅度
      fused   = t_path + s_path + gate*cross

    门控三档（gate_style，训练时对比选最优）:
      score   gate=sigmoid(w_cos*cos + w_mag*mag + bias)       # 3参数，最稳
      mlp     gate=sigmoid(MLP([cos,mag]))                     # 2维小MLP，推荐折中
      full    gate=sigmoid(MLP([cross,s_path,t_path,cos]))     # 最表达力，最易过拟合

    关联强时gate→1，交叉项生效；关联弱时gate→0，静态/时序两条通路各自独立。
    防过拟合: 低秩瓶颈 + 永久独立旁路 + 门控瓶颈（gate_bottleneck）+ dropout
              + 可选融合预算正则（fusion_lambda，由Trainer调用fusion_budget_loss）。
    训练后检查gate分布是否退化（全0/全1=没有学到选择性），并配合固定gate消融。
    训练后|U*V^T|可直接可视化→论文图表素材

    调用: fusion=LowRankCrossFusion(30,5,128,rank=4,gate_style='mlp')
         fused=fusion(static_feat,temporal_feat)
    """
    def __init__(self, static_dim, temporal_dim, hidden_dim, rank=4, dropout=0.1,
                 gate_style='mlp', gate_hidden=8, gate_dropout=0.1,
                 use_feature_relevance=True):
        super().__init__()
        self.rank = rank
        self.use_feature_relevance = use_feature_relevance
        self.U = nn.Parameter(torch.empty(static_dim, rank))
        self.V = nn.Parameter(torch.empty(temporal_dim, rank))
        self.cross_bias = nn.Parameter(torch.zeros(rank))
        nn.init.kaiming_uniform_(self.U)
        nn.init.kaiming_uniform_(self.V)
        if use_feature_relevance:
            # 特征级选择：初始接近全开，模型学习把无对应关系的静态特征关掉
            self.feature_relevance = nn.Parameter(torch.full((static_dim,), 2.2))

        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.cross_proj = nn.Linear(rank, hidden_dim)

        self.gate_style = gate_style
        if gate_style == 'score':
            self.w_cos = nn.Parameter(torch.tensor(1.0))
            self.w_mag = nn.Parameter(torch.tensor(0.1))
            self.gate_bias = nn.Parameter(torch.tensor(-0.5))
        elif gate_style == 'mlp':
            self.gate_net = nn.Sequential(
                nn.Linear(2, gate_hidden),
                nn.GELU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden, 1),
            )
        elif gate_style == 'full':
            gate_in_dim = 3 * hidden_dim + 1  # cross + s_path + t_path + cos
            self.gate_norm = nn.LayerNorm(gate_in_dim)
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden, 1),
            )
        else:
            raise ValueError("gate_style must be one of ['score', 'mlp', 'full']")
        self.gate_dropout = nn.Dropout(dropout)
        self.last_gate = None

    def forward(self, static_feat, temporal_feat, return_stats=False):
        """static[B,S] temporal[B,T,F] -> fused[B,T,hidden_dim]；return_stats=True时返回(fused,stats)"""
        B, T, _ = temporal_feat.shape

        if self.use_feature_relevance:
            s_weighted = static_feat * torch.sigmoid(self.feature_relevance).unsqueeze(0)
        else:
            s_weighted = static_feat
        static_lr = s_weighted @ self.U                        # [B,rank]
        temporal_lr = temporal_feat @ self.V                   # [B,T,rank]
        cross_lr = static_lr.unsqueeze(1) * temporal_lr + self.cross_bias  # [B,T,rank]
        cross = self.cross_proj(cross_lr)                      # [B,T,D]

        s_path = self.static_proj(static_feat).unsqueeze(1).expand(B, T, -1)  # [B,T,D]
        t_path = self.temporal_proj(temporal_feat)             # [B,T,D]

        cos = F.cosine_similarity(
            static_lr.unsqueeze(1).expand(B, T, self.rank),
            temporal_lr, dim=-1
        )                                                        # [B,T]
        mag = cross_lr.sum(dim=-1)                               # [B,T]

        if self.gate_style == 'score':
            score = self.w_cos * cos + self.w_mag * mag + self.gate_bias
            gate = torch.sigmoid(score).unsqueeze(-1)            # [B,T,1]
        elif self.gate_style == 'mlp':
            gate_in = torch.stack([cos, mag], dim=-1)            # [B,T,2]
            gate = torch.sigmoid(self.gate_net(gate_in))         # [B,T,1]
        else:  # full
            gate_in = torch.cat([cross, s_path, t_path, cos.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.gate_net(self.gate_norm(gate_in)))  # [B,T,1]
        self.last_gate = gate

        fused = t_path + s_path + self.gate_dropout(gate) * cross
        if return_stats:
            stats = {
                'gate_mean': float(gate.mean().item()),
                'gate_std': float(gate.std().item()),
                'gate_min': float(gate.min().item()),
                'gate_max': float(gate.max().item()),
                'assoc_mean': float(cos.mean().item()),
                'cross_mag': float(cross.abs().mean().item()),
                'relevance_mean': float(torch.sigmoid(self.feature_relevance).mean().item())
                if self.use_feature_relevance else float('nan'),
            }
            return fused, stats
        return fused

    def fusion_budget_loss(self):
        """融合预算正则: 默认少融合，信号强才放开。由Trainer乘fusion_lambda后加入总loss。"""
        if self.last_gate is None:
            return torch.zeros((), device=self.U.device)
        return self.last_gate.mean()


class CrossConcatMultiHeadAttention(nn.Module):
    """
    交叉拼接多头注意力(CCMA第二阶段)。

    核心: 真交叉注意力。Q来自时序投影，K/V来自低秩融合后的token，
    让“纯时序查询”从“含静态信息的融合表示”中提取交互证据。

    内部=交叉TransformerBlock:
      MultiHeadAttention(Q=temporal, K=V=fused, 8heads) -> 残差+LN -> FFN -> 残差+LN
      附带可学习位置编码（短序列上也能区分时间步）。

    attn_weights[B,T,T]仅用于可解释性；DAFL使用TCN分支注意力branch_attention[B,K]，
    两者不再混用。

    调用: ccma=CrossConcatMultiHeadAttention(128,num_heads=8)
         attended,attn_weights,stats=ccma(temporal_proj,fused)
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, max_len=64):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.position = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, query_feat, fused_feat):
        """query_feat[B,T,D]为时序投影；fused_feat[B,T,D]为低秩融合结果。
        返回 attended[B,T,D], attn_weights[B,T,T], attn_stats dict"""
        B, T, D = query_feat.shape
        q = query_feat + self.position[:, :T]                  # [B,T,D]
        q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(fused_feat).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(fused_feat).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (float(self.head_dim) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, D)
        attended = self.norm1(query_feat + self.out_proj(ctx))
        attended = self.norm2(attended + self.ffn(attended))

        attn_weights = attn.mean(dim=1)                        # [B,T,T]
        eps = 1e-7
        attn_entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
        attn_stats = {
            'attn_mean': float(attn_weights.mean().item()),
            'attn_std': float(attn_weights.std().item()),
            'attn_entropy': float(attn_entropy.mean().item()),
        }
        return attended, attn_weights, attn_stats


# ----- 创新三：膨胀感知动态损失 (DAFL) -----

class DilationAwareFocalLoss(nn.Module):
    """
    膨胀感知动态损失(DAFL) — TCN特有创新。

    核心: 不让模型只盯一个感受野。熵正则改为hinge形式——
    只在分支注意力低于目标熵时才惩罚过度集中，不强制模型均匀使用所有分支。

    loss = focal_loss + lambda*max(0, target_entropy - entropy)

    参数:
      alpha_pos/alpha_neg: 类别权重，建议按违约率设置，可用for_default_rate生成
      gamma_base/gamma_max: gamma调度范围（warmup后才开始上升）
      num_epochs: 总训练轮数
      num_branches: 膨胀分支数(用于max_entropy计算)
      entropy_lambda: 熵正则强度
      entropy_target: 目标熵比例(默认0.8*max_entropy)
      gamma_warmup: gamma保持gamma_base的轮数
      epsilon: 数值稳定性常数

    branch_attention=None -> 退化为标准DynamicFocalLoss

    调用:
      criterion=DilationAwareFocalLoss.for_default_rate(0.22, num_epochs=150)
         criterion.set_epoch(epoch)
         loss,stats=criterion(logits,targets,branch_attention)
    """
    def __init__(self, alpha_pos=0.75, alpha_neg=0.25, gamma_base=1.0, gamma_max=3.0,
                 num_epochs=100, num_branches=4, entropy_lambda=0.1,
                 entropy_target=0.8, gamma_warmup=10, epsilon=1e-7):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_base = gamma_base
        self.gamma_max = gamma_max
        self.num_epochs = num_epochs
        self.num_branches = num_branches
        self.entropy_lambda = entropy_lambda
        self.entropy_target = entropy_target
        self.gamma_warmup = gamma_warmup
        self.epsilon = epsilon
        self.current_epoch = 0

    @classmethod
    def for_default_rate(cls, default_rate, **kwargs):
        """按数据集违约率生成类别权重，避免Taiwan/Home共用一套alpha。"""
        return cls(alpha_pos=1.0 - default_rate, alpha_neg=default_rate, **kwargs)

    def set_epoch(self, epoch):
        """每个epoch开始前调用，更新self.current_epoch用于gamma调度。"""
        self.current_epoch = epoch

    def _gamma(self):
        progress = min(max(
            (self.current_epoch - self.gamma_warmup) /
            max(self.num_epochs - self.gamma_warmup, 1), 0.0), 1.0)
        return self.gamma_base + (self.gamma_max - self.gamma_base) * progress

    def _branch_entropy(self, branch_attention):
        """返回 entropy[N], max_entropy, entropy_gap[N]"""
        p = branch_attention.clamp(min=self.epsilon)
        entropy = -(p * p.log()).sum(dim=-1)
        max_entropy = float(np.log(max(self.num_branches, 1)))
        return entropy, max_entropy, max_entropy - entropy

    def forward(self, logits, targets, branch_attention=None):
        """返回 total_loss(scalar), loss_stats(dict)"""
        gamma = self._gamma()
        prob = torch.sigmoid(logits)
        alpha = torch.where(targets == 1,
                            torch.tensor(self.alpha_pos, device=logits.device),
                            torch.tensor(self.alpha_neg, device=logits.device))
        pt = torch.where(targets == 1, prob, 1.0 - prob)
        focal = -alpha * (1.0 - pt) ** gamma * torch.log(pt + self.epsilon)
        loss = focal.mean()
        loss_stats = {'focal': float(loss.item()), 'gamma': float(gamma)}

        if branch_attention is not None and self.entropy_lambda > 0:
            entropy, max_entropy, _ = self._branch_entropy(branch_attention)
            target = self.entropy_target * max_entropy
            reg = torch.relu(target - entropy).mean()
            loss = loss + self.entropy_lambda * reg
            loss_stats['entropy_mean'] = float(entropy.mean().item())
            loss_stats['entropy_reg'] = float(reg.item())
        return loss, loss_stats


# ----- 完整模型：GAD-TCN -----

class GADTCN(nn.Module):
    """
    门控注意力膨胀TCN — 完整六步流水线。

    1. StaticEmbedding      static[B,S]->Linear->GELU->LN->[B,D]
    2. TemporalProjection   temporal[B,T,F]->Linear->LN->[B,T,D]
    3. TCNEncoder(Parallel) d=1,2,4,8四路+HDG->List[[B,T,D]]x4+branch_attn[B,4]
    4. LowRankCrossFusion   原始static+原始temporal->U*V^T交叉->[B,T,D]
    5. CCMA                 fused->MultiHeadSA->attended[B,T,D]+attn[B,T,T]
    6. Classifier           时序池化+static skip->MLP->logits[B,2]

    模块开关(消融用): use_hdg, use_ccma, tcn_mode
    DAFL在Trainer中控制(use_dafl参数)

    调用:
      model=GADTCN(static_dim=30,temporal_dim=5,temporal_steps=6,
                   hidden_dim=128,num_tcn_layers=4,use_hdg=True,use_ccma=True)
      # 训练
      logits,branch_attn,attn_w,hdg_stats=model(s,t,return_attention=True)
      # 推理
      logits=model(s,t,return_attention=False)
    """
    def __init__(self, static_dim, temporal_dim, temporal_steps, hidden_dim=128,
                 num_tcn_layers=4, kernel_size=3, dilation_rates=None,
                 num_heads=8, lowrank_rank=4, dropout=0.2,
                 use_hdg=True, use_ccma=True, use_fusion=True, use_attention=True,
                 tcn_mode='parallel',
                 gate_style='mlp', gate_hidden=8):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = (1, 2, 4) if temporal_steps <= 6 else (1, 2, 4, 8)
        dilation_rates = list(dilation_rates)
        use_fusion = use_ccma and use_fusion
        use_attention = use_ccma and use_attention
        self.model_config = {
            'static_dim': static_dim,
            'temporal_dim': temporal_dim,
            'temporal_steps': temporal_steps,
            'hidden_dim': hidden_dim,
            'num_tcn_layers': num_tcn_layers,
            'kernel_size': kernel_size,
            'dilation_rates': dilation_rates,
            'num_heads': num_heads,
            'lowrank_rank': lowrank_rank,
            'dropout': dropout,
            'use_hdg': use_hdg,
            'use_ccma': use_ccma,
            'use_fusion': use_fusion,
            'use_attention': use_attention,
            'tcn_mode': tcn_mode,
            'gate_style': gate_style,
            'gate_hidden': gate_hidden,
        }
        self.dilation_rates = dilation_rates
        self.gate_style = gate_style
        self.gate_hidden = gate_hidden
        if use_fusion:
            style = gate_style if gate_style != 'compare' else 'mlp'
            self.cross_fusion = LowRankCrossFusion(
                static_dim=static_dim,
                temporal_dim=temporal_dim,
                hidden_dim=hidden_dim,
                rank=lowrank_rank,
                dropout=dropout,
                gate_style=style,
                gate_hidden=gate_hidden,
            )
        else:
            self.cross_fusion = None
        if use_attention:
            self.attention = CrossConcatMultiHeadAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout)
        else:
            self.attention = None
    def _compute_branch_attention(self, branch_outputs):
        """各分支时间池化->评分->softmax->[B,num_branches]"""
        pass
    def _pool_sequence(self, sequence_out):
        """时序池化: 取最后时间步(因果卷积已含全部历史)"""
        pass
    def forward(self, static_feat, temporal_feat, return_attention=False):
        """训练: return_attention=True; 推理: return_attention=False"""
        pass


# ============================================================
# 五、配置参数
# ============================================================

# 数据集默认超参数
DEFAULT_CONFIGS = {
    'taiwan': {
        'hidden_dim': 128, 'num_tcn_layers': 4, 'kernel_size': 3,
        'num_heads': 8, 'lowrank_rank': 4, 'dropout': 0.2,
        'batch_size': 128, 'epochs': 150, 'lr': 5e-4, 'weight_decay': 2e-4,
        'entropy_lambda': 0.1,
        # 实际数据特征（用于历史长度测试的截断参考）
        'real_temporal_steps': 6,   # Taiwan真实月度
        'temporal_dim': 5,
    },
    'home': {
        'hidden_dim': 128, 'num_tcn_layers': 6, 'kernel_size': 3,
        'num_heads': 8, 'lowrank_rank': 6, 'dropout': 0.25,
        'batch_size': 256, 'epochs': 100, 'lr': 3e-4, 'weight_decay': 1e-4,
        'entropy_lambda': 0.1,
        # 实际数据特征
        'real_temporal_steps': 24,  # Home真实月度
        'temporal_dim': 12,
    },
}

# 非创新模型使用各自主流参数，不与 GADTCN 共用一套超参
BASELINE_ML_CONFIGS = {
    'logistic': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42},
    'random_forest': {
        'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1,
    },
    'svm': {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'},
    'dnn': {
        'hidden_layer_sizes': (256, 128, 64), 'activation': 'relu',
        'alpha': 1e-4, 'max_iter': 500, 'random_state': 42, 'early_stopping': True,
    },
    'xgboost': {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'eval_metric': 'logloss', 'random_state': 42,
    },
    'lightgbm': {
        'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31,
        'class_weight': 'balanced', 'random_state': 42,
    },
}

BASELINE_DL_CONFIGS = {
    'lstm': {'hidden_dim': 64, 'dropout': 0.3, 'lr': 1e-3,
             'epochs': 30, 'batch_size': 128, 'weight_decay': 1e-4},
    'bilstm': {'hidden_dim': 64, 'dropout': 0.3, 'lr': 1e-3,
               'epochs': 30, 'batch_size': 128, 'weight_decay': 1e-4},
    'attention_lstm': {'hidden_dim': 64, 'dropout': 0.3, 'lr': 1e-3,
                       'epochs': 30, 'batch_size': 128, 'weight_decay': 1e-4},
    'resnet_lstm': {'hidden_dim': 64, 'dropout': 0.3, 'lr': 1e-3,
                    'epochs': 30, 'batch_size': 128, 'weight_decay': 1e-4},
}

BASELINE_TCN_CONFIG = {
    'hidden_dim': 64, 'num_tcn_layers': 4, 'kernel_size': 7,
    'dropout': 0.2, 'lr': 1e-3, 'weight_decay': 1e-4,
    'epochs': 50, 'batch_size': 128,
}

# 七指标评估集合（平等做主评估）
PRIMARY_METRICS = ['auc', 'auc_pr', 'brier_score', 'accuracy', 'f1',
                   'sensitivity', 'specificity']

# 消融实验配置（累积激活策略）
ABLATION_CONFIGS = [
    ("Standard TCN",         False, False, False, "stacked"),
    ("+ Parallel TCN",       False, False, False, "parallel"),
    ("+ HDG",                True,  False, False, "parallel"),
    ("+ CCMA",               True,  True,  False, "parallel"),
    ("+ DAFL  Full GAD-TCN", True,  True,  True,  "parallel"),
]

# 不平衡鲁棒性测试的违约率梯度
IMBALANCE_RATES = (0.02, 0.05, 0.10, 0.22, 0.30)

# CCMA 低秩 rank 敏感性实验
RANK_SENSITIVITY_RATES = (2, 4, 6, 8)

# 历史长度测试 —— 按数据集的实际月度截断，两个数据集分别与TCN比较
# Taiwan: 原始6月，测试 1/3/6 月（截取最近N月）
TAIWAN_HISTORY_LENGTHS = (1, 3, 6)
# Home:    原始24月，测试 3/6/12/24 月（截取最近N月）
HOME_HISTORY_LENGTHS = (3, 6, 12, 24)


# ============================================================
# 六、验证实验
# ============================================================

class EarlyStopping:
    """早停机制——验证AUC连续patience轮不提升则停止。"""
    def __init__(self, patience=10, min_delta=0.0001): pass
    def __call__(self, val_auc): pass


class Trainer:
    """
    GAD-TCN训练器——DAFL+EMA+Platt校准+混合精度+七指标评估。

    评估：全部七个指标平等参与。阈值由固定搜索策略确定
    (0.7*F1+0.3*Acc，受Sensitivity下限约束)，确定后所有指标在同一阈值下可比。

    训练组件: AdamW+余弦退火, DAFL/CrossEntropyLoss, AMP+GradScaler,
             梯度裁剪max_norm=1.0, EMA(decay=0.995), Platt交叉校准

    调用: trainer=Trainer(model,train_loader,val_loader,test_loader,
                          num_epochs=150,lr=5e-4,use_dafl=True)
         test_metrics,history=trainer.train()
    """
    def __init__(self, model, train_loader, val_loader, test_loader,
                 calibration_loader=None, num_epochs=100, lr=1e-3,
                 weight_decay=1e-4, use_dafl=True, entropy_lambda=0.1,
                 use_early_stopping=True, threshold_min_sensitivity=0.40,
                 selection_metric='auc_pr', ema_decay=0.995,
                 calibrate_probabilities=True): pass
    def _selection_score(self, metrics):
        """Checkpoint选择分数。按selection_metric计算（默认auc_pr）。"""
        pass
    def _update_ema(self):
        """更新EMA shadow权重：shadow=decay*shadow+(1-decay)*current。"""
        pass
    def _collect_logits(self, loader):
        """遍历DataLoader收集所有logits+labels（Platt校准用）。"""
        pass
    def train_epoch(self, epoch):
        """单epoch训练。DAFL模式: return_attention=True->branch_attn->DAFL。
        返回 avg_loss。"""
        pass
    def evaluate(self, loader):
        """七指标评估。返回dict含: auc, auc_pr, brier_score, accuracy,
        f1, sensitivity, specificity, balanced_accuracy, predictions,
        probabilities, labels。"""
        pass
    def fit_platt_scaling(self):
        """Platt校准: StratifiedKFold交叉拟合+NLL防退化。
        校准参数存入 self.calibration_scale/self.calibration_bias。"""
        pass
    def find_best_threshold(self, min_sensitivity=0.50):
        """阈值搜索: 扫描[0.3,0.6], 评分=0.7*F1+0.3*Acc"""
        pass
    def train(self):
        """完整训练流程 -> 返回 test_metrics, history"""
        pass


# ----- 基线模型 -----

class SequenceBaselineModel(nn.Module):
    """LSTM/Bi-LSTM/Attention-LSTM/ResNet-LSTM基线。
    参数来源(不针对本数据集调参):
      hidden_dim=64  dropout=0.3  lr=1e-3
    调用: model=SequenceBaselineModel(30,5,64,'bilstm')"""
    def __init__(self, static_dim, temporal_dim, hidden_dim=64, model_type='lstm',
                 dropout=0.3): super().__init__(); pass
    def forward(self, static_feat, temporal_feat): pass


def train_sequence_baseline(name, model, Xs_tr, Xt_tr, y_tr, Xs_te, Xt_te, y_te,
                            epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4):
    """训练DL基线，参数来自 BASELINE_DL_CONFIGS。
    返回{acc,auc,auc_pr,f1,sensitivity,specificity}"""
    pass


# ----- 验证实验函数 -----

def compare_baselines(Xs_tr, Xt_tr, y_tr, Xs_te, Xt_te, y_te,
                      deep_epochs=30, batch_size=128):
    """
    实验一：基线对比。

    传统ML: 逻辑回归,随机森林,SVM,DNN,XGBoost,LightGBM（各自主流参数）
    深度学习: Standard LSTM, Bi-LSTM, Attention-LSTM, ResNet-LSTM,
             Standard TCN, Multi-Scale TCN（各自独立参数，不共用GADTCN参数）
    输出七指标对比+GAD-TCN vs 最优基线提升幅度。
    """
    pass


def run_ablation_study(static_dim, temporal_dim, temporal_steps, train_loader,
                       val_loader, test_loader, calibration_loader=None,
                       epoch=50, hidden_dim=128, num_tcn_layers=4,
                       num_heads=8, lowrank_rank=4, dropout=0.2,
                       lr=5e-4, weight_decay=2e-4):
    """
    实验二：消融实验（累积激活策略）。
    5配置依次新增一个模块（非创新TCN用主流参数，创新启用后使用文章模型参数）:
      1. Standard TCN (stacked)
      2. + Parallel TCN (parallel)
      3. + HDG
      4. + CCMA
      5. + DAFL  Full GAD-TCN
    每配置独立训练+七指标评估。输出消融表(行=配置,列=七指标+Δ+参数量)。
    """
    pass


def subsample_training_rate(Xs, Xt, y, target_default_rate, random_state=42):
    """训练集subsample到目标违约率。返回(Xst_sub,Xt_sub,y_sub)"""
    pass


def run_imbalance_robustness(Xs_tr, Xt_tr, y_tr, Xs_v, Xt_v, y_v,
                             Xs_te, Xt_te, y_te,
                             target_rates=IMBALANCE_RATES,
                             epoch=30, batch_size=128, hidden_dim=128,
                             num_tcn_layers=4, lr=5e-4, weight_decay=2e-4):
    """
    实验三：类别不平衡鲁棒性。
    训练集subsample到不同违约率(2%,5%,10%,22%,30%)。
    每档独立训练+七指标评估。DAFL熵正则预期在低违约率下提供更强稳定性。
    """
    pass


def _train_and_evaluate_model(static_train, temporal_train, y_train,
                              static_val, temporal_val, y_val,
                              static_test, temporal_test, y_test,
                              calibration_data=None,
                              epochs=30, batch_size=128, hidden_dim=128,
                              num_tcn_layers=4, use_hdg=True, use_ccma=True,
                              use_dafl=True, lowrank_rank=4, tcn_mode=None,
                              lr=5e-4, weight_decay=2e-4):
    """用给定切分数据训练一个 GADTCN，返回测试集指标。"""
    pass


def _align_static_temporal(static_a, temporal_a, static_b, temporal_b):
    """把两组数据截断到公共静态维度、时间步和通道数。"""
    pass


def run_rank_sensitivity_study(static_train, temporal_train, y_train,
                               static_val, temporal_val, y_val,
                               static_test, temporal_test, y_test,
                               calibration_data=None,
                               epochs=30, batch_size=128, hidden_dim=128,
                               num_tcn_layers=4, lr=5e-4, weight_decay=2e-4):
    """CCMA 低秩 rank 敏感性实验：rank=2/4/6/8，对比效果与参数量。"""
    pass


def run_cross_dataset_study(taiwan_data, home_data, epochs=30, batch_size=128,
                            hidden_dim=128, num_tcn_layers=4,
                            lr=5e-4, weight_decay=2e-4):
    """
    实验四：跨数据集泛化。
    6种组合: Taiwan->Taiwan, Taiwan->Home, Home->Home, Home->Taiwan,
             Joint->Taiwan, Joint->Home
    泛化衰减率=(AUC同域-AUC跨域)/AUC同域。目标<15%。
    """
    pass


def run_history_length_study(static_features, temporal_features, y, dataset_name,
                             history_lengths=None,
                             epochs=30, batch_size=128, hidden_dim=128,
                             num_tcn_layers=4, lr=5e-4, weight_decay=2e-4):
    """
    实验五：历史长度敏感性 —— 按实际月度截取，两个数据集分别与Standard TCN比较。

    每条历史数据均为真实时序记录，不做任何外推或伪造：
      Taiwan:  测试 T = 1, 3, 6  月（从原始6月中截取最近N月）
      Home:    测试 T = 3, 6, 12, 24 月（从原始24月中截取最近N月）

    每个长度下同时训练 GAD-TCN 和 Standard TCN（单分支串行，无创新模块），
    分别报告七指标。两模型对比验证：
      - TCN卷积并行性在长序列上的计算优势
      - HDG的大膨胀分支在T增大时的价值
      - DAFL多感受野协同在长序列下的增益

    输出: 每个数据集的「GAD-TCN vs Standard TCN」对比表
          (行=历史长度, 列=两个模型的七指标)

    Args:
        history_lengths: None时自动选择数据集对应的长度列表
    """
    pass


# ----- 可解释性 -----

def run_shap_analysis(model, data_loader, Xs, Xt, dataset_name, sample_size=None):
    """SHAP可解释性。双层: DeepExplainer->KernelExplainer回退。
    输出Top-20特征(区分静态vs时序贡献)。"""
    pass


# ----- 可视化（PNG 输出；HTML 仪表盘为预留接口）-----

def plot_training_history(history, dataset_name):
    """训练曲线四子图: Loss/AUC/AUC-PR/F1。输出PNG。"""
    pass
def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """混淆矩阵热力图。输出PNG。"""
    pass
def plot_attention_heatmap(attn_weights, dataset_name, sample_idx=0):
    """CCMA注意力热力图[T,T]。输出PNG。"""
    pass
def plot_hdg_lambda_by_layer(hdg_stats_history, dataset_name):
    """HDG各层lambda随epoch变化。输出PNG。"""
    pass
def plot_dilation_entropy(branch_attention, y_true, dataset_name):
    """膨胀熵分布——违约vs正常。输出PNG。"""
    pass
def plot_cross_association_matrix(model, static_names, temporal_names, dataset_name,
                                  ):
    """|U*V^T|低秩交叉关联矩阵热力图[S,F]。输出PNG。"""
    pass
def plot_ablation_summary(ablation_results, dataset_name):
    """消融实验汇总: 七指标分面柱状图。输出PNG。"""
    pass
def plot_cross_dataset_comparison(results_dict):
    """跨数据集泛化AUC柱状图。6组合+衰减率标注。"""
    pass
def plot_history_length_comparison(gadtcn_results, standard_tcn_results,
                                   dataset_name):
    """历史长度对比图: GAD-TCN vs Standard TCN 双折线(AUC为主), 按数据集分面。
    输出PNG。"""
    pass
def plot_rank_sensitivity(results, dataset_name):
    """CCMA rank 敏感性：AUC 与参数量双轴折线。输出PNG。"""
    pass
def build_html_dashboard(model, test_metrics, history, attn_weights, branch_attn,
                         hdg_stats, shap_values, dataset_name):
    """HTML 交互仪表盘为预留接口，本轮不生成。"""
    pass


# ============================================================
# 七、主程序 —— 实验入口
# ============================================================

def run_experiment(dataset_name='taiwan', epochs=None, batch_size=None,
                   threshold_min_sensitivity=0.40,
                   gate_style='mlp', gate_hidden=8,
                   run_baselines=True, run_ablation=True,
                   run_imbalance=True, run_cross_dataset=True,
                   run_history_length=True, run_rank_sensitivity=True,
                   run_shap=True,
                   make_plots=True, save_results=True):
    """
    主实验入口——完整流程。

    1.URL加载数据(Taiwan/Home/Both) -> 数据形状打印+违约率统计
    2.预处理(四层分割+标准化+one-hot)
    3.DataLoader(pin_memory,num_workers)
    4.初始化GADTCN(全模块: HDG+CCMA+parallel, gate_style=用户选择)
    5.训练(DAFL+EMA+Platt校准)
    6.测试集七指标评估
    7.按开关选择性执行:
       若run_baselines     -> 实验一: 基线对比(ML+DL)
       若run_ablation      -> 实验二: 消融实验(5配置累积激活)
       若run_imbalance     -> 实验三: 类别不平衡鲁棒性(5档违约率)
       若run_cross_dataset -> 实验四: 跨数据集泛化(6种组合)
       若run_history_length-> 实验五: 历史长度敏感性(GAD-TCN vs Standard TCN)
       若run_shap          -> SHAP可解释性分析
    8.若make_plots         -> PNG图表
    9.若save_results       -> 保存JSON/checkpoint/bundle到results/

    Args:
        dataset_name:   'taiwan'|'home'|'both'
        epochs:         训练轮数(None=使用默认值)
        batch_size:     批次大小(None=使用默认值)
        threshold_min_sensitivity: 阈值搜索最低敏感性
        gate_style:     CCMA门控样式 score/mlp/full/compare
        gate_hidden:    CCMA门控MLP隐藏维
        run_baselines: 运行基线对比
        run_ablation: 运行消融实验
        run_imbalance: 运行不平衡鲁棒性
        run_cross_dataset: 运行跨数据集泛化
        run_history_length: 运行历史长度敏感性
        run_shap: 运行SHAP
        make_plots: 生成PNG图表
        save_results: 保存结果文件

    Returns:
        model: GADTCN实例
        test_metrics: dict(七指标+threshold+raw arrays)
        history: dict(训练历史 + hdg_stats/dafl_stats)
    """
    pass


def parse_args():
    """CLI 参数解析：只保留基础运行参数，实验开关由交互选择。

    示例:
      python essay_tcn.py
      python essay_tcn.py --dataset taiwan --epochs 200
    """
    parser = argparse.ArgumentParser(description='GAD-TCN Credit Risk Experiment')
    parser.add_argument('--dataset', choices=['taiwan','home','both'], default=None,
                        help='不传则交互选择')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--min-sensitivity', type=float, default=0.40)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    options = interactive_experiment_setup(default_dataset=args.dataset)
    model, metrics, history = run_experiment(
        dataset_name=options['dataset_name'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        gate_style=options['gate_style'],
        gate_hidden=options['gate_hidden'],
        run_baselines=options['run_baselines'],
        run_ablation=options['run_ablation'],
        run_imbalance=options['run_imbalance'],
        run_cross_dataset=options['run_cross_dataset'],
        run_history_length=options['run_history_length'],
        run_rank_sensitivity=options['run_rank_sensitivity'],
        run_shap=options['run_shap'],
        make_plots=options['make_plots'],
        save_results=options['save_results'],
        threshold_min_sensitivity=args.min_sensitivity)

    print(f"\n===== GAD-TCN Final Results =====")
    print(f"AUC-ROC:    {metrics['auc']:.4f}")
    print(f"AUC-PR:     {metrics['auc_pr']:.4f}")
    print(f"Brier:      {metrics['brier_score']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print(f"Sensitivity:{metrics['sensitivity']:.4f}")
    print(f"Specificity:{metrics['specificity']:.4f}")
    print(f"Threshold:  {metrics['threshold']:.3f}")

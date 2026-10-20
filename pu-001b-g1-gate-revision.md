# PU-001B-G1 — 验证契约与授权闭合记录（Gate Revision Record）

**仓库：** `https://github.com/Gay-wushuang/Particle-Universe`
**基线：** `912c0e46e8d00dc285c34258b1113f3a8ea61042`（= 本修订的 HEAD）
**任务性质：** **Docs-only** —— 版本化测试契约与变更授权修订
**状态：** **PU-001B Gate 仍为阻断态**；**未**宣布 PU-001B 完成；**未**宣布 P0 Freeze

> 本记录**不**修改任何 Python 源码、历史原始数据或既有实验结论。`PU-001B.md` 归档的 **`C4 = FAIL` 保持不变**。

---

## 0. 前置核验（只读）

| 项目 | 结果 |
|------|------|
| HEAD | `912c0e46e8d00dc285c34258b1113f3a8ea61042` |
| HEAD 父提交 | `b219f95`（PU-001B 协议提交） |
| 工作区状态 | `git status --porcelain` 为空（清洁） |
| 基线一致性 | 任务给定基线 == HEAD ✓ |
| 远程 | `https://github.com/Gay-wushuang/Particle-Universe.git`，分支 `main` |
| 授权文件 | `pu-000-gate-approval.md`（Decision `APPROVED_FOR_PU001_IMPLEMENTATION`）存在 |
| 现行规范 | `physics-baseline-proposal.md`、`p0-test-matrix.md`、`implementation-handoff.md` 均存在且已读取 |
| PU-001B 归档 | `PU-001B.md`（`STOPPED_ON_C4_FAILURE`）、`artifacts/pu-001b/summary.json` 与 5 个 `*.npz` |
| 双提交只读审计 | PU-001B-C4 Contract & Numerical Audit（会话交付，非仓库文件）；结论摘要见 §1 |

**提交谱系（provenance）：** `ed4faad`（PU-001A 证据基线）→ `b219f95`（PU-001B 协议，含 `potential.py` 改写）→ `912c0e4`（PU-001B 归档，HEAD）。

---

## 1. C4 版本化修订（摘要）

**权威文本：** [`p0-test-matrix.md`](file:///e:/ERE/Documents/Work/Particle%20Universe/p0-test-matrix.md) §3.1（`C4-v2`）。

- `C4-v1` 存在**统计量歧义**（"方差" vs "幅度/标准差" vs 实现所用的"相对标准差"）、**归一化/窗口/平稳性未定义**、**理论适用域被越界使用**、**阈值越权导入**（把 `N1a` 的 `2.0±0.3` 直接用于 C4）、**Gate 类别未声明**，且与 `implementation-handoff.md` §2.5 的 DoD 表（**未列 C4**）不一致。
- `C4-v2` **拆分**：**`C4a`** = 光滑力 `dt²` 涨落阶数（**硬性 Gate**，要求 C¹ 力 → **随机体 B 归 PU-002**）；**`C4b`** = 变体 A 稠密多体涨落**诊断**（归 PU-001，**不断言 `dt²`**，**不阻断 `C5`**）。
- 明确统计量 `σ_E`／`Var(E)`／`r_E`／`ΔE_max`、归一化（主用 `r_E`）、预注册窗口、平稳性/趋势/有界性诊断、去趋势策略（幅度不去趋势；平稳性用去趋势残差，仅诊断）。
- **理论依据（独立）：** 光滑辛积分由影子哈密顿量给出 `σ_E ∝ dt²`、`Var(E) ∝ dt⁴`、`ΔE_max ∝ dt²`、`r_E ∝ dt²`；变体 A 力不连续 → 该律**不适用**。
- **历史不变：** `PU-001B.md` 的 `C4 = FAIL`（`observed_scaling_slope = 1.1840866260516132`）**保持原状**，本修订只重定义**前瞻契约**。

---

## 2. C5 契约闭合（摘要）

**权威文本：** [`p0-test-matrix.md`](file:///e:/ERE/Documents/Work/Particle%20Universe/p0-test-matrix.md) §3.2。

- **实验配置**（`scripts/run_pu001b_validation.py` @ `b219f95`）：**Forward Euler**（显式，teaching-only），**与 C1 同初态**（`multibody_initial_state()`：N=256 FCC、`ρ*=0.8`、`T*=1.0`、`seed=20261020`、PBC、`L=6.8399`），`dt=0.002`、`steps=10000`、`t_end=20`；每步 1 次力评估。
- **比较指标：** 双方同一指标 `max_t |E(t)−E(0)|/|E(0)|`；`ratio = FE_max_drift / VV_max_drift`；代码判据 `ratio ≥ 100` → PASS、`1<ratio<100` → INCONCLUSIVE、`ratio ≤ 1` → FAIL。
- **授权状态：** 冻结文档**仅**写漂移 "**≫**"；实现中的 `ratio ≥ 100` 是**代码默认值，不是规范授权** → 列为**前瞻提案**（见 §5-①）。
- **解耦：** 允许**后续独立执行 C5** 取证，**即使 C4 仍 FAIL**；但 **PU-001B Gate 整体仍阻断**。独立执行须作为**新证据运行**归档，**不得**改写原 `summary.json`／`PU-001B.md`。当前归档中 `C5` 未执行（`not_executed_due_to_gate: ["C5"]`）；独立执行需新协议阶段或独立入口 —— 属**后续任务**（本修订不改 Python）。

---

## 3. 生产代码重构处置（`b219f95` 对 `potential.py` 的向量化）

**事实记录（未事先授权）：**

- `b219f95` 在 PU-001B 协议提交中改写了 [`particle_universe/potential.py`](file:///e:/ERE/Documents/Work/Particle%20Universe/particle_universe/potential.py) 的 `evaluate()`：由逐对循环改为**向量化**（`np.triu_indices` 全对索引 + `boundary.displacement` 广播 + `np.einsum` 求 `r²` + 布尔掩码裁剪 cutoff + `np.add.at` 累加 + `np.flatnonzero` 重叠检测 + `_force_inside_cutoff`/`_energy_inside_cutoff` 向量化核）。
- 该改写**未经事先的 Gate 授权**，且**未附带旧/新数值等价测试**。
- 独立审计（PU-001B-C4 Scope C）结论：**回归风险 LOW** —— 向量化结果与暴力闭式对照在力 ~`1e-15`、能量 ~`1e-14` 内一致；净力和 ~`4.4e-15`；精确 `rc` → 力/能量为 0；`pair_force` 标量路径保持解析正确；PBC/最小镜像、对称性（`np.add.at` 对重复索引安全）、重叠拒绝语义均保持。

**Gate 处置原则（本记录提案）：**

- Gate **原则上允许申请保留**该向量化实现（因其语义等价性已由只读审计证实、风险 LOW）。
- **正式追认的条件：** 在仓库内**新增旧/新数值等价测试**，覆盖以下四项（现行 `tests/` 缺失）：
  1. **向量化 vs 暴力对照** —— 同一随机合法构型（含 PBC）下，`evaluate()` 的 `forces` 与 `potential_energy` 与 O(N²) 双层循环参考实现一致（力 ε≈1e-12、能量 ε≈1e-12，且 `interacting_pairs` 计数一致）。
  2. **标量 `pair_force` 解析值** —— 对给定 `r<rc`，`pair_force` 与该点的解析 `−dU/dr` 一致；`r≥rc` 时恒为 0。
  3. **PBC 下净力 ≈ 0** —— 平移不变性/牛顿第三定律：合法 PBC 构型下 `ΣF ≈ 0`（机器精度量级）。
  4. **精确 `rc` 边界** —— `r == rc` 的粒子对贡献**力=0 且能量=0**（cutoff 语义严格为 `r < rc`）。
- **浮点一致性边界：** **跨提交的浮点计算不保证逐位（bitwise）一致**（求和顺序、向量化归约次序不同）。故等价测试须以**数值容差**判定，**不得**要求逐位相等；任何"逐位一致"主张仅在**同一提交、同一代码路径**下成立。
- **Provenance 边界（须保存）：**
  - 旧实现（逐对循环）：commit `ed4faad`，`particle_universe/potential.py`。
  - 新实现（向量化）：commit `b219f95`，同路径。
  - 归档证据 `artifacts/pu-001b/*` 由 `912c0e4` 记录（HEAD）。
  - 追认测试一经合并，须在提交信息与 `p0-test-matrix.md` §5 元数据中同时标注**测试所针对的实现 commit**。

---

## 4. C1 与证据完整性

- **C1 判据不足（记录）：** C1 的"无单调漂移趋势（仅振荡）"被实现执行为"**非严格单调**"（`strictly_monotonic: false`）。该判据**不足以**排除被振荡掩盖的**带噪长期趋势** —— 已归档 C1 运行虽"非严格单调"，却存在线性趋势 `−5.71e-6`（全程）与首/末 10% 均值相对变化 `−4.03e-6 → −1.02e-5`。
- **替代趋势诊断（提案，见 `p0-test-matrix.md` §3.3）：** 线性拟合 `E(t)=a+b·t` 报告 `b` 及标准误；**Mann–Kendall**（或线性回归 t / Spearman）检验并报告统计量与 p 值；首/末等长块均值对比。**阈值/显著性水平为前瞻提案，须 Gate 核准**，不得以观测值反推。
- **归档增强要求（后续任务，不是本次改动）：**
  1. `N1b` 保存**终态**（`n1b_raw.npz` 现无终态位置/速度）。
  2. `C3` 保存**逐步最大位移**（现仅存净位移 `unwrapped_forward_displacement`）。
  3. `C4b` 保存 **`E/K/U` 分量序列**与**各 `dt` 的截断穿越计数**（现 `c4_raw.npz` 仅存总能量；穿越证据仅在长程运行序列中）。

---

## 5. 前瞻提案清单（须 Gate 核准后方可使用）

> 下列各项**均为实现/设计中的默认或拟定值，尚不构成规范授权**。请 Gate **逐项裁定**为「核准 / 修改 / 驳回」。**禁止**依据已观察数值（`1.1841`、`1.8341` 等）反向选定。

| # | 判据 | 现值/提案 | 独立依据 | 授权状态 |
|---|------|-----------|----------|----------|
| ① | `C5` 漂移比值 | `ratio ≥ 100` → PASS；`1<ratio<100` → INCONCLUSIVE | **定性**有理论（非辛⇒无界漂移 vs 辛⇒有界）；**具体倍数无独立依据** | **待裁定** |
| ② | `C4a` 斜率**中心值** | `r_E`/`ΔE_max` = 2；`Var(E)` = 4 | **有**：反向误差分析／影子哈密顿量 | 中心值可核准；见 ③ |
| ③ | `C4a` 斜率**容差带** | 例 `2.0 ± 0.3` | **无**独立依据（源自 `N1a` 导入） | **待裁定** |
| ④ | 窗口/暂态参数 | 例 `t_relax = 10 τ`、`T_win`、窗口最小步数 | 方法学合理，**数值无独立依据** | **待裁定** |
| ⑤ | 平稳性/趋势检验参数 | `K` 分块数、显著性水平 `α`（Mann–Kendall）、重采样接受带 | **方法**有标准依据；**数值**为设计选择 | **待裁定** |

---

## 6. 待裁定项（保留，不得为关闭测试而任意定义）

以下判据**无法**从独立理论或预注册实验设计中得到充分数值依据，**明确保留为待裁定**：

1. **`C5` 的 `ratio ≥ 100`**（§5-①）—— 保留；如需定量口径，应由 Gate 指定或由**预注册的对照设计**给出。
2. **`C4a` 的斜率容差**（§5-③）—— 保留；建议改为**由块重采样不确定度导出**（预注册），而非沿用导入的 `±0.3`。
3. **窗口/趋势检验的具体数值**（§5-④⑤）—— 保留；须在**运行前预注册**，不得事后调整。

> 依任务要求：**若某新数值判据缺乏独立依据，则保留为待裁定，不得为关闭测试任意定义。** 上述三项即按此处理。

---

## 7. 归档增强要求（后续）

见 §4 尾部的三项；它们**不**要求本次重跑，只在后续证据运行中落实，并回填 `p0-test-matrix.md` §5 元数据。

---

## 8. CI 状态记录（新增，不改写原历史报告）

- **CI run `39364812023` == "Minimal NumPy CI"，`head_sha = 912c0e46e8d00dc285c34258b1113f3a8ea61042`，`conclusion = success`，`event = push`，`run_number = 2`。**
- **覆盖范围：** 仅**快速测试**（`.github/workflows/ci.yml` → `python -m unittest discover -s tests`）。**不**覆盖 PU-001B 长程数值实验（`N1a`/`N1b`/`C3`/`C4`/`C5` 等）。
- 本条为**补充记录**；`PU-001B.md` 中引用的早期 run（`39352681886`）与其它原始历史报告**不改写**。

---

## 9. 历史完整性声明与阻断状态

- **`PU-001B.md` 的 `C4 = FAIL` 保持不变**；其它既有实测结论（`N1a`、`N1b`、`N5`、`C1`–`C3`）**不改写**。
- `C4-v2`（§1）**仅前瞻适用**，不追溯改写历史判定。
- **PU-001B Gate 仍为阻断态**：直至 (i) `C4-v2` 经 Gate 核准；(ii) `C5` 取得独立证据；(iii) `C4a`（PU-002）排期。
- **未**宣布 PU-001B 完成；**未**宣布 P0 Freeze。

---

## 10. 跨文档一致性检查

| 检查项 | 结果 |
|--------|------|
| `C4` 类别（硬性 Gate / 诊断）在矩阵与 handoff 中一致 | ✓ 均为 `C4a` 硬性 / `C4b` 诊断 |
| DoD（§2.5）已含 `C4b` 行、不含 `C4a`（`C4a` 归 PU-002） | ✓ |
| 测试组归属行同步（PU-001 含 `C4b`、PU-002 含 `C4a`） | ✓ |
| `C5` 前瞻标注一致（矩阵 §3.2 与 handoff §2.5/§5.1） | ✓ |
| 归档增强要求一致（矩阵 §5 与本文 §4/§7） | ✓ |
| 理论域声明一致（`physics-baseline-proposal.md` §9.1 与矩阵 §3.1(2)） | ✓ |
| 历史结论未被修改（`PU-001B.md`／`artifacts/`／`PU-001A.md` 未触碰） | ✓ |
| Python 源码未被修改 | ✓ |
| `git diff --check` | 见 §11（须为空/无空白错误） |

---

## 11. 提交记录

- **本修订提交 SHA：** `78e15c6`（"docs: version C4 contract and close PU-001B-G1 authorizations"；4 files changed, 345 insertions(+), 18 deletions(-)）。
- **Diff 摘要：** 改动仅限 Markdown —— `p0-test-matrix.md`（+146/-…）、`implementation-handoff.md`、`physics-baseline-proposal.md`（新增 §9.1）、新增本文件 `pu-001b-g1-gate-revision.md`。**无 Python 源码、无 `artifacts/`、无 `PU-001A.md`/`PU-001B.md` 改动。**
- **记录提交 SHA：** 待回填（本 SHA 由紧随其后的记录提交写入，见 Gate 报告）。

---

## 12. 建议 Gate 裁定

**建议 Decision：** `APPROVED_FOR_C4_V2_CONTRACT_AND_C5_INDEPENDENT_EVIDENCE`

1. **核准** `C4-v2` 契约（§1、§5-② 的中心值），并要求对 §5-③④⑤ 的数值容差**逐项裁定**（或要求改为预注册方案）。
2. **核准** `C5` 与 `C4` 解耦、允许独立取证；**驳回或核准** `ratio ≥ 100`（§5-①）。
3. **有条件追认** `b219f95` 的 `potential.py` 向量化：以新增 §3 的四项旧/新数值等价测试为条件。
4. **保留阻断**：PU-001B Gate 维持阻断；不宣布完成、不宣布 P0 Freeze。

---

## 13. 未决与停止条件

- **未决：** §5-①③④⑤ 的数值判据（保留为待裁定，见 §6）；`C5` 独立执行入口；四项等价测试的实现；三项归档增强。
- **停止条件（未触发）：** 本修订过程中**未**出现无法消除的规范/物理冲突；所有新增定量判据均已在**独立依据（理论锚定）**或**明确保留为待裁定**之间做出处置，未为关闭测试而任意定义阈值。

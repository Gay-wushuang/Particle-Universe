# PU-000 — 物理基线提议书（Physics Baseline Proposal）

**阶段：** P0 — Unit Particle Universe
**状态：** 待 Gate 审查（**不含实现**）
**依赖：** `reference-audit.md`（证据锚定见该文 §0 commit 表）

---

## 1. 目的与范围

本文件定义 P0 第一版**最小求解器**的物理约定：粒子模型、势能、力、运动方程、边界条件、初始化与守恒量定义。

**范围红线（对应任务禁止事项）：**
- 只做 **单组分 Lennard-Jones + NVE**；不引入真实化学、分子反应、3D 世界、ECS、游戏 UI。
- 只做 **NumPy CPU**；不引入 Taichi/JAX/GPU。
- 物理正确性优先于性能；任何加速推迟。

---

## 2. 单位系统（无量纲 / LJ reduced units）

固定基准常量（对应任务 §1）：

| 常量 | 值 | 说明 |
|------|----|------|
| 质量 `m` | 1 | 全部粒子等质量 |
| 长度 `σ` | 1 | LJ 零交叉距离单位 |
| 能量 `ε` | 1 | LJ 势阱深度单位 |
| `k_B` | 1 | 温度以能量表示 |

**注意（任务原文强调）：** `m=σ=ε=kB=1` 是**基准缩放**，并**不**意味着位置、动量、温度或时间步长固定为 1。因此：

- 盒子边长 `L`、温度 `T*`、时间步 `dt` 是**独立可调参数**，需按 §8 取值与验证。
- 由此导出的自然单位：时间 `τ = σ√(m/ε) = 1`，速度 `σ/τ = 1`。

> 对应证据：mdlj-python 的势能直接以 `r⁻¹² − r⁻⁶` 表示、无 σ/ε 参数（`mdsolver/potential.py`），即隐式采用 σ=ε=1；jax-md / taichimd 则以 `sigma`/`epsilon` 参数化但默认 `1`（`jax_md/energy.py` `lennard_jones(sigma=1, epsilon=1)`；`taichimd/examples.py` `ff.set_params(nonbond={1:[1,1]})`）。PU 采用与之一致的无量纲约定。

---

## 3. 粒子模型

| 项目 | 取值 | 备注 |
|------|------|------|
| 维度 | 3 | 与三个代表实现一致（`taichimd/consts.py` `DIM=3`；mdlj README 3D 示例；jax-md LJ 默认 3D） |
| 组分 | 单组分（monodisperse） | 与 mdlj-python / taichimd `lj` 示例一致 |
| 质量 | 全部 `m=1` | 对应 `jax_md/simulate.py` 的 `mass=f32(1.0)` 默认 |
| 粒子数 N | 小规模起步，取 FCC 兼容值（4·cells³）：N ∈ {2, 108, 256, 500} | N=2 解析/可逆对照；N=108/256 多体（配 `rc=2.5`）；**N=500（FCC cells=5）用于 `rc≤3.5` 的敏感性测试**（见 §10、`p0-test-matrix.md` N2） |
| 相互作用 | 仅成对 LJ（无键、无角、无外场） | 对应 tau: taichimd `lj` 示例仅 `nonbond`；`ClassicalFF` 的 bonded/bending/external 在 P0 **不使用** |

**状态变量（最小集）：**
- 位置 `r ∈ ℝ^{N×3}`
- 速度 `v ∈ ℝ^{N×3}`
- 加速度/力 `a, F ∈ ℝ^{N×3}`（`F = m·a`）
- 标量：`t`（时间）、`U`（势能）、`KE`（动能）

> 对照：mdlj-python 存储 `r, v, a, u, n, t`（`mdsolver/__init__.py`）；jax-md 存储 `position, momentum, force, mass`（`jax_md/simulate.py` `NVEState`）。PU 采用**速度**而非动量作为主变量，`m=1` 时二者数值等价。

---

## 4. 势能与截断策略

### 4.1 成对势（标准 12–6 LJ）

$$U_{\text{LJ}}(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right], \qquad \sigma=\varepsilon=1$$

> 证据：jax-md `jax_md/energy.py` → `def lennard_jones` 源码为 `4*epsilon*(idr12 - idr6)`，其中 `idr = sigma/dr`（readthedocs `_modules` 渲染行 272–278）。

### 4.2 截断半径

**推荐 `rc = 2.5 σ`。** 理由：jax-md 成对包装默认 `r_cutoff = 2.5σ`（`energy.py` `lennard_jones_pair`，渲染行 288）；taichimd LJ 示例 `LennardJones(rcut=2.5)`（`taichimd/examples.py`）。

> 说明差异：mdlj-python 默认 `rc = 3σ`（`mdsolver/__init__.py` 写死 `cutoff=3`）。做**交叉对照**时必须统一 `rc`，否则势能/力不可直接比较。

**硬约束**：采用最小镜像时须满足

$$\frac{L}{2} > r_c \quad \text{(每个维度)}$$

否则镜像计数错误。jax-md 对此有显式警告（`jax_md/space.py` → `periodic_general` docstring："uses the minimum image convention, and so it will fail for potentials whose cutoff is longer than the half of the side-length of the box"）。P0 应在初始化时**断言**该条件。

### 4.3 平移 / 平滑策略（两个受支持变体）

**变体 A（P0 v0，默认）：能量平移（truncated & shifted）**

$$U_A(r) = \begin{cases} U_{\text{LJ}}(r) - U_{\text{LJ}}(r_c), & r < r_c \\ 0, & r \ge r_c \end{cases}$$

- 优点：实现最简、可读性最高；力仅由 `U_LJ` 梯度给出（常数平移不影响力）。
- 缺点：**力在 r=rc 处不连续**（`U'_{LJ}(rc) ≠ 0`，外侧力骤降为 0），会引入轻微能量漂移。这是教科书标准做法，也对应 taichimd 的形式（`taichimd/interaction.py` → `LennardJones.__call__`：`4ε[σ¹²(1/r¹² − 1/rc¹²) − σ⁶(1/r⁶ − 1/rc⁶)]`）。

**变体 B（P0 v1，守恒优化）：C¹ 平滑切换（HOOMD / jax-md 风格）**

$$U_B(r) = S(r)\cdot U_{\text{LJ}}(r),\quad S(r)=\begin{cases}1 & r < r_{\text{onset}}\\ \text{（C¹ 三次过渡）} & r_{\text{onset}} \le r < r_c\\ 0 & r \ge r_c\end{cases}$$

- 采用 `r_onset = 2.0σ`, `r_cutoff = 2.5σ`。
- 优点：**力连续**，NVE 能量守恒显著更优。
- 证据（可复用，Apache-2.0）：`jax_md/energy.py` → `def multiplicative_isotropic_cutoff(fn, r_onset, r_cutoff)`，其 `smooth_fn` 为 `(r_c − r)²(r_c + 2r − 3r_o)/(r_c − r_o)³`（渲染行 579–588），且 docstring 明确"follow the approach outlined in HOOMD Blue"。

> **已裁定（PU-000G）**：**变体 A 为 P0 默认**，也是 PU-001 唯一实现的势能变体；**变体 B 延至 PU-002** 作为守恒优化对照（届时以同一积分器、同一边界、同一初始态运行，保证可比）。
> 能量漂移在 PU-001 中仅作**独立诊断**（`p0-test-matrix.md` **N5**）；**收敛阶由 N1 的位置/速度误差给出**，不得以"固定步数终点漂移"的拟合充当收敛阶。

---

## 5. 力

由势能负梯度定义（`m=1` 时力即加速度）：

$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} U = \sum_{j \ne i} \mathbf{F}_{ij},\qquad
\mathbf{F}_{ij} = -\frac{dU}{dr}\Big|_{r_{ij}}\cdot \frac{\mathbf{r}_{ij}}{r_{ij}}$$

对变体 A（rc 内），解析式：

$$\mathbf{F}_{ij} = 24\left(\frac{2}{r_{ij}^{14}} - \frac{1}{r_{ij}^{8}}\right)\mathbf{r}_{ij},\qquad r_{ij} < r_c$$

> 截面校验：`mdsolver/potential.py` → `__call__`：`factor = (2·r⁻¹² − r⁻⁶)/r²`，`force = 24·factor·dr`，等价于上式（`2r⁻¹⁴ − r⁻⁸` 乘 24 乘 `dr`）。可作为实现后的**解析对照**。

**牛顿第三定律**：`F_ji = −F_ij`，仅计算每一无序对的力一次（mdlj-python 用上三角索引实现：`upperTri = np.triu_indices(par, 1)`，`mdsolver/potential.py`）。

**P0 邻域策略**：**不做邻域搜索**，直接全对 O(N²) 计算（与 mdlj-python 一致：`calculateDistanceMatrix` 构造完整 N×N 位移矩阵）。邻域列表/网格分箱推迟（见 §11）。

### 5.1 奇点（r→0）口径 —— PU 不采用 jax-md 的约定

- jax-md 的 `lennard_jones` 以 `jnp.nan_to_num(...)` 包裹，在 `r=0` 时把能量置为 `0`，且源码自带 TODO 注释 "This seems potentially dangerous"（`github.com/jax-md/jax-md/blob/main/jax_md/energy.py` @ `a41c7d1`）。
- **PU 正式实现不采用该口径**：物理上 `r→0` 时 `U→+∞` 必须保持。重叠只能通过**初始化最小间距约束 / 运行时断言 / 显式报错**处理，**不得**静默映射为 0（否则会掩盖未定义状态、污染守恒诊断）。
- 对 `1/r⁶` 的数值处理须**显式、可审计**（在断言保护下计算），不得以"置零/截断"掩盖。

---

## 6. 运动方程与积分方案

### 6.1 连续方程（NVE）

$$m\frac{d^2\mathbf{r}_i}{dt^2} = \mathbf{F}_i,\qquad m=1$$

### 6.2 离散步：Velocity-Verlet（kick-drift-kick）

$$\begin{aligned}
\mathbf{v}_i\left(t+\tfrac{\Delta t}{2}\right) &= \mathbf{v}_i(t) + \tfrac{\Delta t}{2}\,\mathbf{a}_i(t)\\
\mathbf{r}_i(t+\Delta t) &= \mathbf{r}_i(t) + \Delta t\,\mathbf{v}_i\!\left(t+\tfrac{\Delta t}{2}\right)\\
\mathbf{a}_i(t+\Delta t) &= \mathbf{F}_i\big(\mathbf{r}(t+\Delta t)\big)/m\\
\mathbf{v}_i(t+\Delta t) &= \mathbf{v}_i\!\left(t+\tfrac{\Delta t}{2}\right) + \tfrac{\Delta t}{2}\,\mathbf{a}_i(t+\Delta t)
\end{aligned}$$

**每步恰 1 次力评估**；辛、时间可逆、二阶精度。

> 证据：jax-md 的 `velocity_verlet` 正是两步半步动量 + 一次位置 + 重新求力（`jax_md/simulate.py` → `def velocity_verlet`：`momentum_step(dt/2) → position_step(dt) → force → momentum_step(dt/2)`）；taichimd 的 `VerletIntegrator` 结构与之同构（`taichimd/integrator.py`：`prestep` 半移动位置，`poststep` 更新速度 + 半移动位置）。

### 6.3 备选积分器（仅用于对照，不作默认）

| 积分器 | 用途 | 证据 |
|--------|------|------|
| Forward Euler | **非辛反例**（显式，能量单调漂移）——C5 的**唯一**反例 | `mdsolver/integrator.py` `class ForwardEuler` |
| Euler-Cromer | **辛**（半隐式/辛欧拉，适用于可分哈密顿量）——**不得称为非辛**；仅作次要对照，**不作 C5 反例** | `mdsolver/integrator.py` `class EulerCromer` |
| Nose–Hoover NVT | P0 **不使用**（非 NVE） | `taichimd/integrator.py` `class NVTIntegrator`；`jax_md/simulate.py` `nose_hoover_chain` |

**P0 结论：默认且唯一的生产积分器 = Velocity-Verlet。** 依据见 `reference-audit.md` §3。

---

## 7. 边界条件与最小镜像

**推荐：三维周期性边界（PBC）+ 最小镜像。**

$$\mathbf{r}_{ij} \leftarrow \mathbf{r}_{ij} - \mathrm{round}\!\left(\frac{\mathbf{r}_{ij}}{L}\right)\cdot L$$

> 证据：mdlj-python `mdsolver/boundary.py` → `class Periodic.checkDistance`：`dr -= np.round(dr/self.lenbox) * self.lenbox`；jax-md `jax_md/space.py` → `def periodic_displacement`：`jnp.mod(dR + side/2, side) - side/2`（行 150–160）与 `def periodic_shift`（行 178–180）。

**约束与检查：**
- 强制 `L/2 > rc`（§4.2）。
- 位置回卷策略：每步把坐标回卷进 `[0, L)`（mdlj `Periodic.checkPosition` 用 `floor(r/L)*L`；jax-md `periodic_shift` 用 `jnp.mod`）。P0 采用回卷以保证数值稳定。
- **自由边界（free/Open）仅用于单体/二体解析对照**（box 足够大或 `Open`），避免镜像干扰。证据：`jax_md/space.py` `def free`；`mdsolver/boundary.py` `class Open`。

---

## 8. 初始化方案

### 8.1 位置

1. **FCC 晶格**（多体默认）：按密度 `ρ*` 生成面心立方，对应 mdlj-python `mdsolver/initposition.py` `class FCC(cells, lenbulk)`（README 示例 `FCC(cells=6, lenbulk=10.2)`）。
2. **随机均匀**（对照）：避免重叠（最小间距 > 0.9σ），或用软核松弛（P1 再议）。

### 8.2 速度（Maxwell–Boltzmann）

- 每分量 `v ∼ 𝒩(0, k_B T / m)`，`σ_v = √(T*)`。
- **质心动量归零**：`v ← v − mean(v)`，消除整体漂移。
  > 证据：jax-md `initialize_momenta` 对多粒子做 `p = p - jnp.mean(p, axis=0, keepdims=True)`（`jax_md/simulate.py`）。
- **温度换算（统一口径）**：`KE = ½ Σ m|v|²`，`k_B = 1`，且 **N>1**：

  $$T^{*} = \frac{2\,KE}{3N - 3}$$

  分母 `3N−3` 即**总自由度**（3N 减去 3 个质心平动自由度），其中**已含维度因子 3**。**不得**再乘维度 `d=3`——否则得到 `2KE/(9N−9)`，属**维度重复计数**。
  > **N=1 边界**：无质心可扣，此时自由度 = 3，`T* = 2KE/3`；本方案的温度统计只用于 `N>1`（N=2 的 M2 亦满足）。
  > **对照（PU 不采用）**：taichimd `NVTIntegrator.integrate_thermostat` 用 `2·ek − 3·N·T` 形式，即自由度取 `3N`、**未扣质心**（`taichimd/integrator.py` @ `57c02b9`）。

### 8.3 确定性

- 所有随机数由**显式种子**驱动，记录 seed + 参数 + commit 以保证复现（见 `p0-test-matrix.md` **R1**）。

---

## 9. 守恒量定义（诊断口径）

| 量 | 定义 | 期望行为 |
|----|------|----------|
| 总能量 `E` | `KE + U`，其中 `KE=½Σ m|v|²`，`U=Σ_{i<j} U_{A/B}(r_ij)`（**含截断口径**） | NVE 下有界振荡，无长期漂移 |
| 总动量 `P` | `Σ m·v` | 严格守恒（PBC + 对力）；机器精度内为常量 |
| 温度 `T*` | `2·KE/(3N−3)`，`k_B=1`，仅 `N>1`（见 §8.2） | 平衡后在初值附近涨落 |
| 位力/压力 | P1 再议（P0 不强制） | — |

> 对照：mdlj-python 的 `thermo` 输出 `poteng/kineng/temp`（`mdsolver/thermo.py`、`mdsolver/__init__.py` `thermo()`）；jax-md `quantity.kinetic_energy` / `quantity.temperature`（`jax_md/simulate.py`）。

**⚠️ 能量口径警示**：跨实现对照时，势能必须统一截断/平移口径。mdlj-python 的平移项用 `N` 而非 `N_pairs`（`reference-audit.md` §2），故**不得**将其绝对势能直接用作基准。

---

## 10. 默认参数与开放决策

**建议默认值（待 Gate 确认）：**

| 参数 | 建议值 | 依据 |
|------|--------|------|
| `dim` | 3 | 参考实现一致 |
| `σ, ε, m, k_B` | 1 | 无量纲基线 |
| `rc` | 2.5 σ | jax-md / taichimd 默认 |
| 势能变体 | **A（能量平移）为 P0 默认（已裁定）**；B 延至 PU-002 | §4.3 |
| 积分器 | Velocity-Verlet | §6 |
| `dt`（**候选值**） | 0.002 τ（**仅候选**，非默认上限） | 介于各参考 0.005–0.01；须由 **N1** 收敛测试确认，安全上限未定 |
| `ρ*` | 0.8 | LJ 液相典型稳定密度 |
| `T*`（初值） | 1.0 | 常见 LJ 流体工况，避免低温结晶干扰 |
| `N`（多体） | 守恒测试 N=256（FCC cells=4）；`rc≤3.5` 敏感性测试 **N=500（FCC cells=5）** | 兼顾 O(N²) 成本、统计与合法盒长 |
| 盒子 | `L = (N/ρ*)^{1/3}`，断言 `L/2 > rc` | §4.2；见下方盒长核算 |

**盒长合法性核算（PU-000G）：**
- `rc=2.5`、`N=256`、`ρ*=0.8`：`L = (256/0.8)^{1/3} = 6.840` → `L/2 = 3.420 > 2.5` ✓
- `rc=3.5`、`N=500`、`ρ*=0.8`：`L = (500/0.8)^{1/3} = 8.550` → `L/2 = 4.275 > 3.5` ✓（FCC cells=5 → 4·5³ = 500）
- 该断言须在初始化时执行；不满足即**拒绝运行**，不得静默继续。

**决策状态（PU-000G 部分裁定）：**
1. ✅ **已裁定**：P0 默认 **变体 A**；变体 B **延至 PU-002**。
2. ⏳ **仍开放**：`dt` 的最终默认值与安全上限——须由 **N1**（位置/速度误差二阶收敛）与 **N5**（能量漂移诊断）确定；`dt=0.002 τ` **仅候选**。
3. ✅ **已裁定**：变体 A/B 对照报告归入 **PU-002**。

---

## 11. 明确不做（P0 范围外）

- 邻域列表 / 网格分箱 / cell list（来源示例：`jax_md/partition.py` `cell_list`、`taichimd/grid.py` `Grid`）——**推迟**。
- Taichi / JAX / Numba / GPU 加速 ——**推迟**（`reference-audit.md` §5）。
- NVT / NPT 温压耦合（`taichimd/integrator.py` `NVTIntegrator`；`jax_md/simulate.py` `nose_hoover_chain`）——**推迟**。
- 多组分、键/角/二面角、外场、刚性体、patchy 粒子（`hoomd-examples` 06/07 教程涉及）——**推迟**。
- 任何 3D 渲染 / 游戏化 / ECS —— **禁止**。
- 任何源自 particle-life 的"规则力"模型 —— **禁止**作为物理验证依据。

---

## 12. 与参考实现的对应关系（可追溯映射）

| 本方案要素 | 主要参照 | 证据（URL + 路径 + commit） |
|-----------|----------|------------------------------|
| LJ 12–6 形式 | jax-md | `github.com/jax-md/jax-md/blob/main/jax_md/energy.py` @ `a41c7d1` |
| rc=2.5σ | jax-md / taichimd | `jax_md/energy.py` `lennard_jones_pair`；`taichimd/interaction.py` + `taichimd/examples.py` @ `57c02b9` |
| 能量平移截断 | taichimd / mdlj-python | `taichimd/interaction.py` `LennardJones.__call__`；`mdsolver/potential.py` `potentialEnergy` @ `0540ac5` |
| C¹ 平滑切换（变体 B） | jax-md（源自 HOOMD） | `jax_md/energy.py` `multiplicative_isotropic_cutoff` @ `a41c7d1` |
| 力 = 势能负梯度 + 牛顿第三定律 | mdlj-python | `mdsolver/potential.py` `__call__`（上三角技巧） @ `0540ac5` |
| Velocity-Verlet | jax-md / taichimd / mdlj-python | `jax_md/simulate.py` `velocity_verlet`；`taichimd/integrator.py` `VerletIntegrator`；`mdsolver/integrator.py` `VelocityVerlet` |
| PBC + 最小镜像 | mdlj-python / jax-md | `mdsolver/boundary.py` `Periodic`；`jax_md/space.py` `periodic`/`periodic_shift` |
| FCC 初始化 | mdlj-python | `mdsolver/initposition.py` `FCC` @ `0540ac5` |
| 速度初始化 + 质心归零 | jax-md | `jax_md/simulate.py` `initialize_momenta` |
| 温度口径 `2KE/(3N−3)`（`N>1`；PU 统一） | 自定（对照见 taichimd / jax-md） | 见 §8.2；对照 `taichimd/integrator.py` @ `57c02b9`（DOF=3N，未扣质心）与 `jax_md/simulate.py` @ `a41c7d1` |

---

## 13. 本文件的验证责任与修订记录

- 本方案**尚未**被任何实测验证（只读审计阶段）。所有数值阈值、`dt` 上限、守恒验收标准均由 `p0-test-matrix.md` 定义并在 PU-001/PU-002 中实测，**不得**在本阶段声称已通过。
- **修订记录（PU-000G / Gate Corrections）**：
  1. 温度口径统一为 `T* = 2KE/(3N−3)`（`N>1`），消除维度重复计数（§8.2、§9、§12）。
  2. 裁定 **变体 A 为 P0 默认**，变体 B **延至 PU-002**；`dt=0.002 τ` 仅为**候选值**（§4.3、§10）。
  3. 明确 **Euler-Cromer 为辛积分器**，非辛反例仅 Forward Euler（§6.3）。
  4. 新增 **§5.1 奇点口径**：PU 不采用 jax-md 的 `jnp.nan_to_num` 约定。
  5. 清理过时测试编号：`M5 → N1`、`M8 → R1`。

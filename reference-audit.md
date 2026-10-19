# PU-000 — 参考项目与技术基线审计报告

**阶段：** P0 — Unit Particle Universe
**任务类型：** Research / Read-only Audit
**文档性质：** 只读审计报告，不含任何实现代码
**审计日期：** 2026-10-20
**重要声明：** 本报告所有结论均基于对公开仓库源码/文档的**实际读取**。凡涉及"验证/测试"的内容，一律标注为**待执行计划**，不声称已运行。

---

## 0. 证据基线（Commit 冻结）

本报告全部技术判断均锚定到下列 commit，避免"最新版本漂移"导致结论失效。

| # | 仓库 | 默认分支 | 审计 Commit (SHA) | Commit 日期 | 许可证 |
|---|------|----------|-------------------|-------------|--------|
| 1 | `github.com/evenmn/mdlj-python` | `master` | `0540ac5aca3c8f4b75f4ef0114943ea766bfeb2a` | 2024-04-12 | setup.py 声明 MIT，**仓库内无 LICENSE 文件** |
| 2 | `github.com/jax-md/jax-md` | `main` | `a41c7d19f6468f4e5263c32c12c9ed6cba26ebff` | 2026-08-18 | Apache-2.0 |
| 3 | `github.com/glotzerlab/hoomd-examples` | `trunk` | `d8af856be90dc0882d9916475366989a8abb499e` | 2026-10-14 | BSD-3-Clause |
| 4 | `github.com/victoriacity/taichimd` | `master` | `57c02b90315a8b14872849f101d14b2eedaa5f66` | 2022-05-26 | **GPL-3.0** |
| 5 | `github.com/hunar4321/particle-life` | `main` | `256278714c4f6a1ce900d24faafcc101769c54c2` | 2024-08-29 | MIT |

证据获取方式：`raw.githubusercontent.com`（源码原文）、GitHub 公开 API `/contents`（目录清单）、`commits/<branch>.atom`（提交 SHA）、readthedocs `_modules`（jax-md `energy.py` 全文）、GitHub license 页面。

> 注：GitHub 公开 API 存在未认证速率限制（60 次/小时，共享出口 IP 曾触发 403 `API rate limit exceeded`）。目录清单与许可证结论均已通过至少一条非 API 渠道复核。

---

## 1. 审计任务 A — 各项目如何组织六个核心子系统

### 1.1 mdlj-python

**证据文件：** `mdsolver/__init__.py`、`mdsolver/potential.py`、`mdsolver/integrator.py`、`mdsolver/boundary.py`（均位于 commit `0540ac5`）

| 子系统 | 实现方式 | 证据（文件 → 位置） |
|--------|----------|---------------------|
| 粒子状态 | `MDSolver` 持有 `r, v, a, u, n, t` 六个 ndarray/标量字段；`numparticles × numdimensions` 由 `r.shape` 推出 | `mdsolver/__init__.py` → `class MDSolver.__init__` |
| 势能函数 | `LennardJones(Potential)` 类，构造时缓存 `forceShell = np.zeros((N,N,dim))` 与上三角索引 `upperTri` | `mdsolver/potential.py` → `class LennardJones.__init__` |
| 力计算 | 构造**完整 N×N 距离矩阵** `drAll`，取上三角，仅保留 `r² < rc²` 的项；`force = 24 * factor * dr`，再按索引回填 `forceShell` 并 `sum(axis=1)` | `mdsolver/potential.py` → `calculateDistanceMatrix` / `__call__` |
| 积分器 | `VelocityVerlet`（默认）、`ForwardEuler`、`EulerCromer`，统一签名 `__call__(r, v, a) → (r, n, v, a, u)` | `mdsolver/integrator.py` |
| 边界条件 | `Open` / `Reflective` / `Periodic` 三个类；PBC 用**最小镜像**：`dr -= np.round(dr/lenbox) * lenbox` | `mdsolver/boundary.py` → `class Periodic.checkDistance` |
| 邻域搜索 | **无**。全对距离矩阵 O(N²)，靠 numpy 向量化，文档明确承认"性能无法与低级语言包竞争" | `mdsolver/potential.py`；`README.md` |

关键设计事实：
- `MDSolver.__init__` 内**硬编码** `self.potential = self.LennardJones(self, cutoff=3)`，即截断半径写死为 3σ（`mdsolver/__init__.py`）。
- 势能采用 **σ=1、ε=1 的无量纲形式**（代码直接写 `r⁻¹² − r⁻⁶`），与 PU 基线单位一致。

### 1.2 jax-md

**证据文件：** `jax_md/simulate.py`、`jax_md/energy.py`、`jax_md/space.py`、`jax_md/partition.py`（commit `a41c7d1`）

| 子系统 | 实现方式 | 证据（文件 → 位置，行号为 raw 渲染行号≈真实） |
|--------|----------|--------|
| 粒子状态 | `NVEState` dataclass：`position, momentum, force, mass`；`velocity` 为只读 property = `momentum/mass`。全部为 JAX PyTree，可 JIT/vmap | `jax_md/simulate.py` → `class NVEState` |
| 势能函数 | `lennard_jones(dr, sigma=1, epsilon=1)` 返回 `4ε((σ/r)¹²−(σ/r)⁶)`（用 `jnp.nan_to_num` 包裹） | `jax_md/energy.py` → `def lennard_jones`（readthedocs `_modules` 渲染行 252–278） |
| 力计算 | **不手写力**：`quantity.canonicalize_force` 对能量函数取 `jax.grad`（自动微分）得到力，再在 `velocity_verlet` 中前向求值 | `jax_md/simulate.py` → `nve()` 内 `force_fn = quantity.canonicalize_force(...)` |
| 积分器 | `velocity_verlet` 原语（kick-drift-kick）；`nve()`；`nvt`/`npt` 基于 Nose–Hoover chain + Suzuki–Yoshida 权重 | `jax_md/simulate.py` → `def velocity_verlet` / `def nve` / `nose_hoover_chain` |
| 边界条件 | `space.free()` / `space.periodic(side)` / `space.periodic_general(box, fractional_coordinates)`；返回 `(displacement_fn, shift_fn)` 二元组 | `jax_md/space.py` → `def free` / `def periodic`（行 182/194）/ `def periodic_general`（行 242） |
| 邻域搜索 | `CellList` / `cell_list`（均匀网格，XLA 静态形状）与 `neighbor_list` | `jax_md/partition.py` → `class CellList`（行 59）、`def cell_list`（行 234） |

关键设计事实：
- 空间被抽象为**位移函数 + 移动函数**对，势能只接收"位移/度量"，因此边界条件与势能完全解耦。
- `periodic_general` 文档明确警告：**使用最小镜像，当势能截断超过盒子半边长时失效**（`jax_md/space.py` → `periodic_general` docstring）。

### 1.3 hoomd-examples

**证据文件：** `README.md`、`01-Introducing-Molecular-Dynamics/00-index.ipynb`、`01-Introducing-Molecular-Dynamics/01-Molecular-Dynamics-Simulations.ipynb`（commit `d8af856`）；根目录含 `00-…` 至 `11-Active-Particles-In-MD` 共 12 个教程目录（GitHub API `/contents?ref=trunk` 确认）。

**关键结论：本仓库本身不是求解器，而是 HOOMD-blue 的 Jupyter 教学材料。**
- 真正的状态/势能/力/积分/边界/邻域搜索全部位于**外部 `hoomd` 包**（不在本仓库内）。
- 本仓库内容以 notebook 形式演示如何调用：`hmd.md.Integrator(dt=0.005)`（`01-Molecular-Dynamics-Simulations.ipynb` 代码单元），并以 **Lennard-Jones 体系结晶**为主线（`00-index.ipynb` 原文："using the Lennard-Jones system as an example"）。
- notebook 的讲点包含"HOOMD-blue 如何在截断半径内计算 pair potential、不同系综的积分方法"，即 **教学方法论**可借鉴，但源码不可直接复用（其本质是调用外部 C++/CUDA 引擎）。

### 1.4 taichimd

**证据文件：** `taichimd/system.py`、`taichimd/interaction.py`、`taichimd/integrator.py`、`taichimd/examples.py`、`taichimd/consts.py`、`run_examples.py`（commit `57c02b9`）

| 子系统 | 实现方式 | 证据 |
|--------|----------|------|
| 粒子状态 | `Simulation` 用 Taichi field 组织：`position, velocity, type, force, ep, ek`；隐式积分额外分配 `hessian`（N×N 矩阵 field） | `taichimd/system.py` → `class Simulation.__init__`（persisted raw，行≈20–60） |
| 势能函数 | `LennardJones(PairInteraction)` 等，接口以 `r²` 与参数元组 `args` 定义；`ClassicalFF(nonbond=…)` 组合非键/键/弯角/外场 | `taichimd/interaction.py` → `class LennardJones`；`taichimd/forcefield.py`（`ClassicalFF`） |
| 力计算 | 由势能的一阶导数手写：`force = 2 * derivative(r²) * r`；同时提供 `second_derivative`（hessian） | `taichimd/interaction.py` → `class PairInteraction.force` / `hessian` |
| 积分器 | `ForwardEulerIntegrator`、`VerletIntegrator`、`NVTIntegrator`（Nose–Hoover，双链 Q1=Q2=5）、`ImplicitMidpoint/BackwardEuler`（Jacobi / FixedPoint 求解器） | `taichimd/integrator.py` |
| 边界条件 | `Boundary` 枚举：`FREE=0, PERIODIC=1, REFLECTIVE=2` | `taichimd/system.py` → `class Boundary` |
| 邻域搜索 | `taichimd/grid.py` 的 `Grid`，示例中通过 `use_grid=True` 启用（`ljmixture`/`biglj` 使用） | `taichimd/__init__.py` → `from .grid import Grid`；`taichimd/examples.py` → `use_grid` 参数 |

关键设计事实：
- 所有示例默认 `ti.init(arch=ti.cuda)`，强依赖 GPU（`run_examples.py`）。
- `consts.py` 注释：`DIM = 3`，"代码未在 3 以外维度测试过"（`taichimd/consts.py`）。
- 参数以 `sigma`/`epsilon` 的 6/12 次幂预计算缓存（`fill_params`），并按 Lorentz–Berthelot 混合：`σ_ij=(σ_i+σ_j)/2`、`ε_ij=√(ε_i ε_j)`（`taichimd/interaction.py` → `class LennardJones.combine`）。

### 1.5 particle-life

**证据文件：** `README.md`（内嵌 JS 核心算法）、`particle_life/src/ofApp.cpp`（C++ 核心，commit `2562787`）

- 状态：`atoms[]` 每个元素 `{x, y, vx, vy, color}`；分组 `group[]`。
- 力：**规则驱动，非势能驱动**。`ofApp.cpp` → `ofApp::interaction(...)`：`force = 1/sqrt(d²)`（仅当 `d < radius`），`fx += (dx) * force`；`g = G / -100`。
- 积分：`v = (v + f) * 0.5; x += v`（显式、无 dt、与真实物理单位无关，JS 版同理）。
- 边界：越界速度取反（`vx *= -1`）。
- 邻域：无，O(N²) 双循环（README "Todos" 第 3 条自述其为性能瓶颈）。

**结论：该项目不含任何可迁移的物理求解器语义**，其价值仅在"参数规则涌现复杂图案"这一**观感**层面。

---

## 2. 审计任务 B — Lennard-Jones 实现、截断与平移/平滑策略

| 项目 | LJ 形式 | 截断半径 | 平移/平滑策略 | 证据 |
|------|---------|----------|---------------|------|
| mdlj-python | `4(r⁻¹² − r⁻⁶)`（σ=ε=1 硬编码） | **rc = 3**（写死在 `MDSolver.__init__`） | **能量平移**：`shift = par·(rc⁻¹² − rc⁻⁶)`，`U = 4·(Σu − shift)`。⚠️ 见下方缺陷 | `mdsolver/potential.py` → `LennardJones.__init__` / `potentialEnergy`；`mdsolver/__init__.py` |
| jax-md | `4ε((σ/r)¹²−(σ/r)⁶)` | **r_cutoff = 2.5σ**（成对包装默认值） | **乘性平滑切换**：`multiplicative_isotropic_cutoff(lennard_jones, r_onset=2.0σ, r_cutoff=2.5σ)`，构造 **C¹ 连续** 的 `S(r)`：`S=1 (r<r_onset)`、`S=0 (r>r_cutoff)`，`f'(r)=S(r)·f(r)`；**其做法明确注明"follow the approach outlined in HOOMD Blue"** | `jax_md/energy.py` → `def lennard_jones`（渲染行 252–278）、`def lennard_jones_pair`（渲染行 282–304，`r_onset=2.0`/`r_cutoff=2.5`）、`def multiplicative_isotropic_cutoff`（渲染行 548–594，`smooth_fn` 行 579–588） |
| hoomd-examples | 由外部 `hoomd.md.pair` 提供 | 由 notebook 设定 | 未在本仓库内实现；jax-md 引用的 HOOMD 文档即为其平滑切换来源 | `01-…/01-Molecular-Dynamics-Simulations.ipynb`（pair potential within a cutoff）；jax-md `energy.py` 注释引用 `hoomd-blue.readthedocs.io/en/stable/module-md-pair.html` |
| taichimd | **能量平移**：`U = 4ε[σ¹²(1/r¹² − 1/rc¹²) − σ⁶(1/r⁶ − 1/rc⁶)]`，即 `U_LJ(r) − U_LJ(rc)` | `rcut` 参数：LJ 体系 `rcut=2.5`；真实单位的 chain/propane `rcut=14` | 硬截断 + 能量平移（位移与力连续，**力不连续**于 rc） | `taichimd/interaction.py` → `class LennardJones.__call__`（`u = 4e(s12(1/r2^6 − irc12) − s6(1/r2^3 − irc6))`）；`taichimd/examples.py` → `LennardJones(rcut=2.5)` |
| particle-life | 无势能 | `radius`（像素） | 无 | `particle_life/src/ofApp.cpp` → `interaction()` |

### ⚠️ 缺陷标注（mdlj-python 能量平移项）

`mdsolver/potential.py` 中：

```
self.shift = par * (cutoff**(-12) - cutoff**(-6))     # par = N（粒子总数）
...
return 4 * (np.sum(u) - shift)
```

严格的能量平移应为 `4·Σ_pairs[U(rc)]`，即 `4·N_pairs·(rc⁻¹² − rc⁻⁶)`，其中 `N_pairs` 为**截断内实际配对数**。而代码使用 `N`（粒子总数）。二者一般**不相等**，因此 mdlj-python 报告的"总势能"在截断处并非严格平移量。

**影响范围：**
- 该差异只影响**势能的常数偏移**，不影响逐对力（平移量为常数，梯度为零），故**不影响 NVE 轨迹**。
- 但会使"用该库的势能绝对值作为能量守恒对照基准"产生系统性偏差。

**审计结论：** mdlj-python 适合作**结构/教学参考**与**轨迹/力对照**，但**不适合**将其实测总势能值作为 P0 的绝对能量基准。

---

## 3. 审计任务 C — Velocity-Verlet 与其他积分方法对比

各项目**实际存在**的积分器（均为读源码确认）：

| 项目 | 可用积分器 | 证据 |
|------|-----------|------|
| mdlj-python | `ForwardEuler`、`EulerCromer`、`VelocityVerlet`（默认） | `mdsolver/integrator.py` |
| jax-md | `velocity_verlet` → `nve`；Nose–Hoover chain（Suzuki–Yoshida 1/3/5/7 阶）→ NVT/NPT；另有随机积分器（`stochastic`）与提交 `a41c7d1` 引入的 NPT CSVR | `jax_md/simulate.py` → `velocity_verlet`, `nve`, `nose_hoover_chain`；提交标题 "Add reference NPT CSVR integrator (#413)" |
| hoomd-examples | 通过 `hmd.md.Integrator` 选择（HOOMD 引擎内部实现，如 NVE/NVT） | `01-…/01-Molecular-Dynamics-Simulations.ipynb`（`dt=0.005`） |
| taichimd | `ForwardEuler`、`Verlet`、`NVT`（Nose–Hoover）、`ImplicitMidpoint`、`BackwardEuler`（Jacobi / FixedPoint） | `taichimd/integrator.py` |

**对比（定性）：**

| 方法 | 辛性/时间可逆 | 精度阶 | 每步力评估 | 适用性 |
|------|---------------|--------|-----------|--------|
| Forward Euler | 否 | 1 | 1 | 能量单调漂移，不适合 NVE 生产 |
| Euler-Cromer | 半隐式，有界但非严格辛 | 1 | 1 | 教学可用，长期守恒不佳 |
| **Velocity-Verlet** | **是（辛、时间可逆）** | **2** | **1** | **NVE 标准选择**，能量有界振荡无长期漂移 |
| Nose–Hoover（NVT） | 扩展哈密顿 | 2 | 1 | 控温，P0 不需要 |
| 隐式（Jacobi/FixedPoint） | 需迭代收敛 | 2 | 迭代多次 | 刚性体系，成本高，P0 不需要 |

**P0 推荐：Velocity-Verlet（kick-drift-kick）。** 理由：
1. 辛且时间可逆 → NVE 下**能量有界振荡而无系统性漂移**，是"守恒性验证"最干净的积分器；
2. 二阶精度，误差 ∝ dt²，便于做**时间步长收敛**（收敛曲线斜率可预测）；
3. 每步仅 1 次力评估，最小求解器实现成本最低；
4. mdlj-python / jax-md / taichimd **三者均以 Velocity-Verlet 作为其 NVE 主力**（互为交叉印证），便于后续对照。

---

## 4. 审计任务 D — NVE 测试与复现方式调研

| 项目 | 测试/复现设施 | 证据 | 可借鉴点 |
|------|---------------|------|----------|
| mdlj-python | 仅 `examples/` 目录 + `analyze.Log`（读 `thermo` 日志、算 RDF）。**未见单元测试目录** | `mdsolver/analyze.py` → `class Log`；`mdsolver/__init__.py` → `write_rdf`；README "For more examples, see the examples folder" | 用**日志文件**记录 `temp/poteng/kineng` 时间序列，便于事后守恒性绘图 |
| jax-md | `pyproject.toml` 声明 `testpaths = ["tests"]`，且 README 强调跨 CPU/GPU/TPU 的一致性 | `jax-md/pyproject.toml` → `[tool.pytest.ini_options]` | 需"**同一实现跨后端结果一致**"的思路（对我们即"参考实现 vs 加速实现"一致性） |
| hoomd-examples | 教程内以**可视化 + 定量曲线**（结晶、RDF 等）演示 | `00-…` 至 `11-…` 教程目录；`02-Logging` | 日志/量化输出的**教学范式** |
| taichimd | 示例脚本 `run_examples.py`；未见测试目录 | `run_examples.py` | 直接可跑的最小示例 |
| particle-life | 无 | — | — |

**审计结论：** 五个项目**均未提供**可直接照搬的"NVE 能量/动量/时间步长收敛"自动化测试套件。P0 必须**自建**测试矩阵（见 `p0-test-matrix.md`），并把 mdlj-python / jax-md 作为**独立第二实现**做交叉对照，而非把它们当作"已通过的测试"。

---

## 5. 审计任务 E — NumPy CPU 参考实现 vs Taichi 加速实现

| 维度 | NumPy CPU（mdlj-python 范式） | Taichi（taichimd 范式） |
|------|-------------------------------|------------------------|
| 可读性/可验证性 | 高：公式与代码近一一对应，易逐步核对 | 低：`@ti.func/@ti.kernel` 语义、field 内存模型、GPU 隐式并行，调试需转换思维 |
| 数值透明度 | 高：float64 默认 | 中：示例大量用 `f32`，且 `NVTIntegrator` 混用 f32/f64（`ti.cast(..., ti.f32)`），需注意精度 |
| 性能 | O(N²) 且在 CPU 上，万粒子量级即吃力（mdlj 自述不如低级语言） | 高：`biglj` 演示 0.26M 粒子（需 ≥6 GB 显存 CUDA） |
| 依赖 | 仅 `numpy`（+`tqdm`） | `taichi`（且示例写死 `ti.cuda`），需要 GPU |
| 可移植性 | 高（无 GPU 亦可） | 低（无 CUDA 时示例直接失败） |
| 许可证风险 | 无（但 mdlj 无 LICENSE 文件，见 §6） | **GPL-3.0，实现不可复制** |
| 维护状态 | 2024 年仍更新 | 2022-05 最后一次提交，绑定 taichi 1.0.1 |

**最小技术依赖方案（P0 推荐）：**
- **P0 仅依赖 NumPy**（必要时加标准库），**不引入 Taichi / JAX / Numba / PyTorch**。
- 原因：(1) P0 目标是**物理正确性**而非性能；(2) 粒子数 N 小（数百级），O(N²) 足够；(3) 依赖越少，数值行为越可控、越易审计；(4) 避免 GPL（taichimd）与重型依赖链（jax-md 的 pyproject 依赖含 jax/flax/jraph/pymatgen/e3nn-jax/equinox 等，见 §6）污染基线。
- 加速（Taichi/Numba/GPU/邻域列表）明确**推迟到 P0 之后**，且必须先有可对照的 NumPy 参考实现。

---

## 6. 审计任务 F — 许可证、依赖、兼容性风险与可复用范围

### 6.1 许可证矩阵与复用边界

| 项目 | 许可证 | 可复用范围（法律层面） | 风险 |
|------|--------|------------------------|------|
| jax-md | **Apache-2.0** | 可复用源码（需保留版权/许可声明、注明修改；含专利授权） | 低（许可）；**高（依赖重量）** |
| hoomd-examples | **BSD-3-Clause** | 可复用 notebook/文档（需保留声明；不得用 Glotzer lab 名义背书衍生品） | 低；但其价值是**教学方法**，非求解器源码 |
| particle-life | **MIT** | 可复用源码（需保留声明） | 低（许可）；**但无物理价值** |
| mdlj-python | setup.py 声明 **MIT**，但**仓库内无 LICENSE 文件**（GitHub license API 返回 `license: null`；`blob/master/LICENSE` 返回 404） | **法律状态不明确**：仅元数据声明，缺少许可文本与版权归属文件 | **中–高**：作为参照可读，但**不建议逐行复制其代码**进入正式基线 |
| taichimd | **GPL-3.0** | ⚠️ **不可复制实现**：copyleft 会传染至衍生作品 | **高**：与用户"不复制 GPL 项目实现"的红线完全冲突 |

> **注**：`evenmn/mdlj-python` 的许可证结论为"元数据声明 MIT 但无 LICENSE 文件"。这是"声明"而非"已确认的法律事实"，需在 Gate 复核时二次确认（见 §8 未决问题）。

### 6.2 依赖与兼容性风险

| 项目 | 主要依赖 | 兼容性风险 |
|------|----------|-----------|
| mdlj-python | `numpy`, `tqdm`（`setup.py`） | 极低；纯 CPU、无编译 |
| jax-md | `absl-py, numpy, jax>=0.5.0, jaxlib, flax, jraph, einops, ml_collections, e3nn-jax, equinox, optax, frozendict, pymatgen>=2021.3.4, typing-extensions`；要求 Python ≥ 3.10；版本 `0.2.29` | 高：依赖树庞大且与 jax 版本强耦合（pyproject 明确排除 `jax==0.10.2`/`0.11.1`）；作为 P0 依赖不合适，仅适合作为**离线对照** |
| hoomd-examples | `fresnel, freud, gsd, hoomd, matplotlib, pillow, row, signac`（conda-forge） | 需 conda 环境 + 外部 HOOMD 引擎；不适合作为 P0 运行时依赖 |
| taichimd | `taichi`（示例 `arch=ti.cuda`）、`taichi_glsl`、`taichi_three`（内置精简版） | 高：写死 CUDA、绑定 taichi 1.0.1、2022 后停更 |
| particle-life | 无（C++ 版需 openFrameworks；JS 版零依赖） | 低；但仅供观感参考 |

### 6.3 可直接复用范围汇总

- **可复用源码（须署名）**：jax-md（Apache-2.0）作为**数学对照实现**的参考；其中 `multiplicative_isotropic_cutoff` 的 C¹ 切换公式与 `periodic_general` 的最小镜像语义可直接借鉴（Apache-2.0 允许）。
- **可复用文档/方法**：hoomd-examples（BSD-3-Clause）的教程结构与"截断半径内 pair potential"讲解范式。
- **仅可借鉴思想（不可复制代码）**：taichimd（GPL-3.0）—— 可借鉴其"势能导数/二阶导数并列接口""Grid 邻域搜索""Nose–Hoover 双链"的**设计思想**，但**不得复制实现**。
- **仅供观感参考（无物理语义）**：particle-life。
- **可读但不宜复制**：mdlj-python（MIT 声明但缺 LICENSE 文件）。

---

## 7. 审计任务 G — 源码参考 vs 思想借鉴 的明确划分

| 项目 | 定位 | 理由（带证据） |
|------|------|----------------|
| **jax-md** | ✅ **数学正确性对照（首选）** | 有 `tests/`（`pyproject.toml`）、Apache-2.0、力由 `jax.grad` 自动微分（`simulate.py` `canonicalize_force`）、LJ 平滑截断有明确 C¹ 公式（`energy.py` `multiplicative_isotropic_cutoff`）、PBC/最小镜像语义清晰（`space.py`）。适合作为**独立第二实现**做交叉数值对照。 |
| **mdlj-python** | ✅ **结构/教学参考（次选）** | 类结构清晰（MDSolver/Potential/Integrator/Boundary 分层），纯 NumPy，与 PU 的"最小依赖"方向一致；但需注意其能量平移项缺陷（§2）与缺 LICENSE 文件（§6）。 |
| **hoomd-examples** | 🔶 **方法论/教学参考** | 本身非求解器（`README.md`）；可借鉴教程组织与"截断内 pair potential"讲法。 |
| **taichimd** | 🔶 **仅借鉴思想（GPL）** | GPL-3.0（`master/LICENSE`）；可借鉴势能导数接口、Grid、Nose–Hoover 设计，**实现不可复制**。 |
| **particle-life** | ⚠️ **仅观感参考** | 规则力模型（`ofApp.cpp` `interaction()`），无势能/守恒语义；**不得**将其图案涌现当作物理守恒的证据（对应任务禁止事项第 6 条）。 |

---

## 8. 尚未确认的问题（需 Gate 复核）

1. **mdlj-python 许可证**：`setup.py` 声明 MIT 但仓库内无 LICENSE 文本。需确认上游是否存在其他许可声明（如 README/issue），或**默认按"不可复制、仅可读"处理**。
2. **jax-md `lennard_jones` 的 `jnp.nan_to_num` 语义**：源码 `energy.py` 中 `return jnp.nan_to_num(4ε(idr12−idr6))` 且带 TODO 注释 "This seems potentially dangerous"。在 r→0 时被置 0 而非 +∞。作为对照实现时需确认其是否影响近距离能量对照。
3. **hoomd-examples 的截断/切换细节**：其 LJ 截断与切换由**外部 `hoomd` 包**决定，本仓库未含实现，故无法从本仓库给出其确切 `r_onset/r_cutoff` 数值（jax-md 引用了 HOOMD 文档作为来源）。
4. **各项目"最小可用版本"**：未在本环境实际安装运行（本任务为只读审计），因此**性能/数值行为为文档与源码推断，非实测**。
5. **对冲时间步长区间**：各项目默认 dt 不同（mdlj `dt=0.01`；hoomd 教程 `dt=0.005`；taichimd `dt=0.01`），无量纲化到 PU 基线后的具体安全区间需 P0 实测确定（见 `p0-test-matrix.md`）。

---

## 9. 风险登记

| 风险 | 等级 | 说明 | 缓解 |
|------|------|------|------|
| 误将 taichimd 实现复制进基线 | 高 | GPL-3.0 copyleft | 明确红线：**只借鉴思想**；code review 时禁止逐行搬运 |
| 误将 mdlj-python 视为"权威能量基准" | 中 | 能量平移项用 N 而非 N_pairs（§2） | 仅用其**轨迹/力/结构**做对照，不用其绝对能量 |
| 把 particle-life 图案当作物理验证 | 中 | 其模型无势能/守恒（§1.5） | 在文档与测试中**显式排除**其为验证依据 |
| 过早引入 Taichi/JAX 依赖 | 中 | 依赖重、与 jax 版本强耦合、CUDA 硬依赖 | P0 仅 NumPy；加速推迟（§5） |
| 最小镜像失效 | 中 | 截断 > 盒子半边长时 PBC 错误（jax-md `periodic_general` 明确警告） | P0 强制约束 `L/2 > rc` 并在初始化时断言 |
| 依赖未认证 API 导致证据链断裂 | 低 | GitHub API 速率限制 | 关键结论均有 raw/atom/readthedocs 备用来源 |

---

## 10. 完成标准自查（回答任务 §6 的 5 个问题）

1. **第一版最小求解器需要哪些模块？**
   → 5 个必需模块：粒子状态容器、LJ 势能（含截断）、力计算（由势能梯度）、Velocity-Verlet 积分器、PBC 最小镜像边界；外加**能量/动量诊断**与**确定性随机数种子**。详见 `physics-baseline-proposal.md`。

2. **哪个参考项目最适合作为数学正确性对照？**
   → **jax-md**（Apache-2.0，有 tests，力由自动微分得到，截断公式明确；`testpaths=["tests"]` in `pyproject.toml`）。次要对照：mdlj-python（仅力/轨迹）。

3. **哪些验证必须先于多粒子实验？**
   → 先做**单体/二体解析对照**与**力=势能负梯度有限差分校验**（M1–M2），再做**能量/动量守恒**（M3–M4）与**时间步长收敛**（M5）。多粒子现象学（RDF、结晶等）必须置于其后。详见 `p0-test-matrix.md`。

4. **哪些性能优化可以推迟？**
   → 邻域列表 / 网格分箱、Taichi/Numba/GPU 加速、JAX/JIT 化、并行化 —— **全部推迟**（§5）。

5. **PU-001 与 PU-002 是否具备无歧义的执行边界？**
   → 是。边界与验收要求见 `implementation-handoff.md`；两者共享同一测试矩阵，PU-002 不新增物理特性，只做验证与加速前置评估。

---

## 11. 下一步建议

- **Gate 复核**：确认 §8 的 5 项未决问题（尤其 mdlj-python 许可证与 jax-md 对照口径）。
- 通过后，按 `implementation-handoff.md` 启动 **PU-001**（最小 NVE 求解器）与 **PU-002**（验证），并以 `p0-test-matrix.md` 为准绳。
- **不进入实现阶段**，等待最终 Gate 审查。

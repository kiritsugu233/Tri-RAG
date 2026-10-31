# Status

Updated 2026-11-01. Maintained branch: `codex/tls-rag-step5-ann`.
Canonical checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.

## 方案4 P1恢复与初次科学冻结（2026-11-01）

- 用户实际指示：“继续P1”，恢复已呈示的第34节八路径范围；旧暂缓决定仍作历史保留。
- 新增五个隔离文件、测试入口单项归类及窄幅登记已完成；不改旧科学源/配置/依赖。
- 新增13项CPU/offline回归及完整442项通过，0跳过，72.162秒；收集Step5=244、legacy=198。
  资源超限真实子进程停机/封存、无真值计时接口、独立池、损失/计数、PQ实际API均覆盖。
- 初测保留两项发现：浮点除法再乘回不应要求逐位还原；当前系统拒绝killpg但允许终止
  自建子进程PID。现验证共同Gaussian原始抽样与已追踪PID树停机，未放宽科研阈值。
- 下一项为本地CPU真实P1，唯一计划run为`runs/route4-screen-20261101-v1`，20k/96历史
  已开放查询，科学累计45分钟、4GiB RSS/3GiB新产物，原cert/test保持关闭。
  尚无新真实数据运行结果；不训练顾问/adapter、不做1M/SSH/方案3/5或多视图。
- 现有四项未提交L1方案5记录保持原字节内容；本轮只将其完整前缀保留并追加第36节。
  保护240项、旧235项等级/旧科学哈希、差异/文档与来源检查在提交前核对。

## 方案5记录：离线建图接口（2026-10-31）

- 用户实际要求：“把这个作为方案5写进去，你检查一下repo中用于存Step5的md文件”。
- 已将讨论追加到[探索日志第35节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#35-方案5离线建图时的邻居选择与剪枝2026-10-31)：
  离线HNSW邻居选择/剪枝，区分研究采样与真正建图，拟用Gaussian比较/精算回退，
  最终仍原空间查询。新增参数/比较器、条件风险及图质量桥梁均未验证。
- 记录R5-A成本筛查→固定现场→便宜比较器→真建图→外部查询验证；完整费用、
  已知精确A的强对照、自适应选择与尾部错误、停止条件和后续审批要求均明确。
- 当前状态planned/not_run；只改四项L1文档及导航，不改源码/测试/配置/保护登记，
  不开启新数据或实验。原日志完整保留，方案4“保留P0，暂缓P1实现”决定不变。
- 文档检查：`python3 scripts/check_code_protection.py`、`git diff --check`，
  并核对旧日志完整前缀、四路径L1范围、本地链接/新锚点及代码围栏。
  全部通过：235项保护登记、53个本地链接、新第35节锚点及围栏；旧日志完整字节保留。
  当前产物是本地设计记录，未提交/推送；没有新的建图结果，未运行429项运行时套件。
- 下一步仍需用户决定具体研究/实现范围；本次记录不自动执行方案5或恢复其他路线。

## 方案4后续研究与 Step 5 交接（2026-10-30）

- P0已完成并追加[第33–34节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)：
  独立核对2013内积符号翻转律与当前距离反转边际的代换、候选计数界、共同投影反例、
  SRS/PM-LSH及压缩基线差异、可用输入与成本接口。378组表达式核对最大绝对差
  `4.551914400963142e-15`；2880项小集合检查通过。这不是新检索算法验收。
- P0证据目录：`runs/route4-p0-review-20261030-v1`。只校验既有100k向量/ID与768条
  历史已开放tune导出包身份，未抽新子集、检索、建索引、训练或访问封闭角色。
- 用户本轮实际答复：“保留P0，暂缓P1实现”。P0结论保留，P1暂缓，不自动恢复。
- 第34节为未批准/暂缓的P1精确提案：本地CPU、20k/96条历史查询、独立IVF候选池、
  Gaussian/PCA/PQ与完整原空间对照、免费M上限、45分钟科学运行上限。
  包含五个新文件、测试入口单项归类、人工/机器登记的窄幅变更；尚未批准或执行。
  不因P0数学成立而声称界足够紧、系统有收益或整篇论文无新意。
- 用户实际指示：“可以，把下一轮研究内容写进方案4的后续里面，然后交付给Step5的agent让它继续按这个路线进行下去”。
- 已追加[探索日志第32节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)，
  保留第21节原方案与全部路线1/2历史结果；更新启动入口及里程碑。
- 主线为离线联合`(m,M)`配置＋独立候选池上的Gaussian批量筛选；
  同预算独立视图并集为条件后续。接手聊天：“TLS-RAG Step 5 ANN 实验方案”。
- 首项P0独立复核与新文件/API、角色/输入、参数/资源及验证的L0提案已完成。
  已有研究方向指示无需重复确认；尚未给出的精确受保护实现范围仍按AGENTS审批。
- 当前产物是研究/交接文档。方案4仍无ANN运行、训练、索引构建或新数据访问；
  6000组公式表达核对属于此前只读分析，不是方案4算法验证。
- 文档检查命令：`python3 scripts/check_code_protection.py`、`git diff --check`，
  并核对原日志为新日志的完整前缀、仅五项L1文档变更及本地Markdown链接/围栏。
  本轮全部通过：235项保护登记、56个本地链接；旧登记及科学L0内容未变。
  本次不重跑运行时套件，不将文档校验写成新算法通过。
- 本轮交付至P0结束；P1、后续P2→P3→P4均不自动执行。等待新的用户指示与相应
  精确范围批准。不自动回到旧HNSW逐节点过滤/停止接法，不开启原方案3或旧封闭角色。

## 已执行的测试精简方案 B（2026-10-30）

- [中文审批方案及430方法逐项表](docs/TEST_SUITE_SIMPLIFICATION_PROPOSAL.md)。
- 用户实际批准：“执行”，承接该方案四路径范围；父提交`7b9d44b8c85bbfb964afcaaf8fdbd1c2485ae299`。
- 230项方法保留、2项等价合并为1项、198项退出日常且完整保留。永久删除0项。
- 无参数/`full`完整429；`step5`日常231；`legacy`历史198。实际收集互斥且并集完整。
  部分套件打印未选范围，不能记作全仓通过；所有实现交付前仍必须完整回归。
- M01四场景保留；三个临时错误接受故障均被检出。其余45文件及同文件其他五方法原样。
- 全集收集/导入错误、未知/缺失测试文件、重复ID、未知套件和多余参数验收均会失败；
  仓库外调用及无参数/full等价核对通过。没有新增仓库测试模块或测试方法。
- 固定依赖/四原生库，从`/private/tmp`无参数运行：429/429通过，0跳过，65.411秒。
  显式Step5：231/231通过，0跳过，39.611秒；历史：198/198通过，0跳过，25.973秒。
  缓存测试单独6/6通过；两次部分运行均明确报告完整验证为false。
- 六组科研来源绑定（55/22/13/8/7/10项）保持一致；科学源/配置/依赖及旧结果未变。
  原235项等级/理由和批准历史保留，仅两个获批L0内容哈希及本次实际批准追加。
- 精确重放命令、历史子集结果、日志SHA和兼容限制见[方案B验收](docs/TEST_SUITE_SIMPLIFICATION_PROPOSAL.md#八实际批准与实施验收2026-10-30)。
- 当前产物为现有启动器、等价合并测试及登记/交接文档；用户未跟踪档案保持原样。
  该测试精简检查点原待用户指示；当前研究交接见顶部。进一步永久删减、科研算法/
  部署安全修复仍需独立范围批准。

## 已批准的方案 A：保护检查与登记修复（2026-10-30）

- 实际批准：“批准执行，执行完毕后更新github上的repo，不要另起branch，直接在step5的分支上提交”。
- 父提交 `61b92d7871ca3da703c137abb4d49cdf9eca487e`；继续 Step5 分支。
- 修复 C/C++/头文件和 ignored mechanism 漏检；补齐人工表 44 项，已有等级不变。
- 原 10 个治理测试方法保留，仅新增 1 个参数化方法，覆盖 30 个原生文件/Git 状态组合；
  ignored Python 覆盖扩展为五个目录，并验证被忽略的载荷目录不被扫描。
- 四个获批 L0 文件范围、批准记录和验收见
  [方案 A](docs/REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair)。
  原失败证据保留；科学代码/测试/配置/依赖不变。五组 Step5 绑定一致；
  A800 来源清单含保护脚本，其身份随获批编辑改变，旧包按原提交复现，不刷新旧哈希。
- 原检查器执行最终新回归可复现失败；修复后治理测试 11/11 通过。
- 完整固定环境命令：`TRI_RAG_TEST_PYTHON=/private/tmp/step5-ann-enhancement-env-20261028/bin/python sh scripts/run_tests.sh`，
  同时设置审查记录列出的四个 STEP5 原生库环境变量；430/430 通过，无跳过，64.757 秒。
- `python3 scripts/check_code_protection.py` 通过，234 项；人工表完整，
  仅三个获批哈希变化。日志/失败证据及验收记录已保存。
- 该A检查点尚未批准/执行方案B；后续独立批准与执行见本文顶部。A的原验收保持历史含义。

## 上一轮：仓库结构审查与整理


- Review baseline: `50defc35bc9f42f5b24d2dedf5f45a89e51257ad`.
  [Review/approval proposal](docs/REPOSITORY_CLEANUP_REVIEW.md),
  [file/dependency map](docs/REPOSITORY_MAP.md),
  [evidence](docs/repository_cleanup_evidence.json).
- Shortened startup/status/plan navigation; retained detailed history in the fixed
  pre-cleanup Git snapshot. Corrected the outdated next-agent state and retired
  Step 4 branch instruction. Added a repository landing page and lifecycle map.
- Test launcher accepts `TRI_RAG_TEST_PYTHON` and resolves the repository from its
  own location. Suite, assertions and native tests are unchanged.
- No runtime source/config/test/dependency removed or modified; no L0 hash refreshed.
  Old experiments still bind those paths. User archives/report_work/runs preserved.
- Confirmed checker blind spots for new native files/ignored mechanism code;
  human protection register omits40 machine-registered paths. Exact four-path
  protected repair was subsequently approved and implemented as A above.
- Baseline default environment:429 tests,4 missing-FAISS errors,33 skips.
  Existing fixed-version/native environment:429/429 pass,0 skips (66.634s).
  Final launcher from `/private/tmp`:429/429 pass,0 skips (65.151s). Protection234,
  all61 Markdown checks and six unchanged scientific binding maps pass; receipts
  are in the review and evidence file.
- That checkpoint next requested approval for A, now received/executed above. No new
  retrieval, policy fitting, data access, cluster run or scientific claim follows.

## Latest scientific checkpoint — continuation v2

Scientific code `08ed4ce`; exact prior consent:
“批准第26节全部范围，按A→B→C执行”.
[Journal sections27–31](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)
hold the full process, commands, source/selection/model checks and limitations.

- Historical development:768 queries,4 nested folds,5 timing repeats.
  All768 FAISS/boundary resume checks pass;429 offline tests passed at that milestone.
- Full original ANN=1.168709ms; S=1.248047/G=1.393090/T=1.629399/
  T0=1.657511/selectiveTri=1.415616ms. Current learned/Tri paths fail net gain.
- T retention=.9971354 vs original=.9984375; its delta interval crosses−.002
  tolerance, with one extra5/10-neighbor tail loss. No trained-support/OOD
  abstention fallback; execution is complete, deployment safety acceptance is not.
- Preserve resumable backend and failures; no automatic retry/threshold change,
  adapter/1M/routes3/4, SSH/GPU or closed-role access. No fresh independence or
  formal certification. Route4 remains planned; route3 has not run.
- Latest raw evidence: `runs/ann-continuation-v2-20261030-v1`,1658 previously
  verified files/2,149,374,244B. This cleanup did not re-audit or rewrite them.

## Historical records

Detailed status/approvals/commands from all previous stages are retained in
[STATUS at the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md), including Step 4 replay,
Step 5 cloud/local storage receipts and original failures. Retrieve offline with:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:STATUS.md
```

The unchanged [exploration journal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)
and versioned contracts remain local. [Milestone index](docs/IMPLEMENTATION_PLAN.md)
links the historical plan. Historical text describes its original authorization;
read current AGENTS and the user's task before acting.

## Branch-retirement provenance — 2026-10-23

The eight remote references below were previously retired with user approval
and ancestry/open-PR checks. This cleanup performs no additional branch deletion.
Their complete original execution receipt is in the fixed status snapshot above
and `docs/code_protection.json`.

| Retired GitHub branch | Retained tip (full SHA in protection registry) |
|---|---|
| `codex/calibrated-tri-predict-v2` | `26079ce67793844f649d7df16ce818c3b539529f` |
| `codex/calibrated-tri-predict-v3` | `f94c1aa63df459b4d2281e296f400942a8552c00` |
| `codex/tls-rag-step3` | `3913ded9b786d6b6b15c734c750bd13d9c58e12c` |
| `codex/tls-rag-step4-readiness` | `fc38d928e428dc8c10a3ea16e5ec5db5ccfbf72a` |
| `codex/tls-rag-step4-real-probe` | `c08cd3b6800888c8d2a5a5f8c58a77d883a84ede` |
| `codex/tls-rag-step4-retention-v3` | `4f836375c8e45361ef7e3229e66f2c1735abee1b` |
| `codex/tri-law-sequential-rag-step2` | `f46ce73beccf5fddbf05fd87fdb0911318020add` |
| `codex/tri-law-sequential-rag-v1` | `cac654ed73f75db92a1d11c09d10e9cd9973a37f` |

# 技术方案与模块 2 验收

## 核心链路

使用 LangChain 的工具定义和模型适配，使用 LangGraph 的 StateGraph 连接两个节点。模型绑定工具时要求每轮返回工具调用，并关闭并行调用；模型仍自行选择具体工具和参数：

1. 模型节点读取用户问题、历史消息及上一工具的实际返回，只提交一个动作：搜索、读取原文或结束。
2. 执行节点校验动作名和参数，调用本地只读知识库，校验搜索或读取结果，将实际结果作为 ToolMessage 送回模型。
3. 模型再次选择动作，直至提交结束动作或触及上限。

搜索排序只提供候选证据，不决定答案。程序路由只处理动作分发、输入校验与执行限制，不按关键词或工具状态预设模型下一步。每次运行最多执行指定数量的模型决策步和资料工具调用；达到上限返回 limit_reached。

## 边界和状态

- 模型决策：生成工具调用和结束提案。
- 工具执行：search_documents、read_document 使用模块 1 的合成资料；错误按结构化结果返回。
- 程序控制：校验动作和工具返回、计数、路由、记录 Trace。
- 在模块 2 的阶段验收中，模型提出 completed 时只返回 pending_validation。模块 3 已加入机械引用核验，当前运行通过后可返回 completed；语义支持仍需人工评估。其他结束提案保留 needs_clarification、insufficient_evidence 或 failed。
- 模型调用异常、无效动作、无效输入、保存 Trace 失败及到达上限均不返回成功。

Trace 的每项包含运行 ID、时间、模型类型、阶段、动作及参数、实际工具结果、执行状态。工具事件在下一轮模型决策后补充 next_action。不记录模型内部思维。

## 运行与验证

依赖版本由 uv.lock 锁定。将 .env.example 复制为本地 .env，并填写 DASHSCOPE_API_KEY 与 DASHSCOPE_BASE_URL；默认模型为 deepseek-v4.1-flash-1017，可用 DASHSCOPE_MODEL 覆盖。.env 已列入 .gitignore。供应商兼容性必须通过实际调用验证，单纯成功创建模型客户端不算通过。

在项目根目录运行：

~~~powershell
uv sync --extra dev
uv run --extra dev pytest -q
uv run knowledge-agent "报销期限是多久？" --trace examples/live-trace.jsonl
~~~

带对话历史时，可用 --history-file 指向一个 JSON 数组，每项包含 role（user 或 assistant）和 content。无效历史由输入校验返回 invalid_input。

tests/test_agent.py 使用明确标记为 test_double 的模拟模型，只验证图的状态流转、工具结果传递、错误与限额。真实模型运行使用用户提供的 deepseek-v4.1-flash-1017 与北京区域接口，凭据仅在被 Git 忽略的本地 .env 中。

2026-10-29 的模块 2 阶段验收结果（历史记录；当前 examples/ 已在模块 3 重新运行）：

| 场景 | 实际状态 | 观察到的关键路径 |
| --- | --- | --- |
| S01 正常 | pending_validation | 搜索 → 读取原文 → 提交有引文的回答提案 |
| S02 冲突 | needs_clarification | 搜索 → 读取 D01、D02 → 再查证 → 请求确认 |
| S03 资料不足 | insufficient_evidence | 首次无结果 → 改写检索仍无结果 → 说明缺口 |
| S06 受控故障 | failed | 搜索错误 → 模型再次尝试仍错误 → 明确失败 |
| S07 步数上限 | limit_reached | 第一步后停止 |
| S08 工具次数上限 | limit_reached | 第一次工具调用后停止 |
| S09 受控首轮空结果 | pending_validation | 人为首轮无结果 → 模型改写检索 → 读取原文 → 提交回答提案 |

模块 2 阶段共运行 54 项离线测试，全部通过；CLI 也完成一次真实模型运行。S06、S09 的工具观察是受控注入，不代表真实资料召回表现。当前 examples/ 的运行结果与模块 3 验收见 docs/03-evaluation.md。

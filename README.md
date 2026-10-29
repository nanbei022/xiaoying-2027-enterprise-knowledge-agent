# 企业知识问答 Agent

小影科技 AI Agent 笔试题目 3 的分阶段实现。核心链路为：模型选择搜索、读取或结束动作 → 程序校验并执行 → 模型观察实际结果后决定下一步。资料均为合成内容；当前保存的 20 个场景状态与预期一致，不能据此推断真实企业资料上的准确率。

## 环境与安装

- Python 3.11 或更新版本
- 安装 uv，以使用 uv.lock 中锁定的依赖

~~~powershell
uv sync --extra dev
~~~

## 本地测试

~~~powershell
uv run --extra dev pytest -q
~~~

测试不需要模型 API Key。当前 82 项测试通过，包含 20 组已保存运行材料的一致性检查；测试替身只验证程序流程，不作为真实模型效果证据。

## 真实模型运行

将 .env.example 复制为 .env，在本机填入 DASHSCOPE_API_KEY 和你自己的 DASHSCOPE_BASE_URL。默认模型为 deepseek-v4.1-flash-1017，可用 DASHSCOPE_MODEL 覆盖。项目不会在可提交文件中保存工作区专属接口地址或密钥。模型必须支持工具调用。

~~~powershell
Copy-Item .env.example .env
# 编辑 .env 中的 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL
uv run knowledge-agent "正式员工申请年假需要提前多久？" --trace logs/manual-trace.jsonl
~~~

模型提交 completed 提案后，程序会检查引用的文档和段落是否已实际读取、引句是否为原文的连续子串。通过后才返回 completed；不通过则把错误交回模型，在剩余额度内修正。CLI 会打印最终状态、回答、引文和 Trace。此机械检查不能替代对“引文是否真的支持结论”的人工评估。

项目已使用真实模型运行全部 20 个合成场景。输入、实际输出与 Trace 位于 examples/，可用以下命令复跑；复跑会重新调用模型并覆盖已有示例文件：

~~~powershell
uv run python examples/run_live_cases.py --all
~~~

S06 是受控工具故障，S09 是首轮搜索被人为置为空结果；这些场景验证真实模型如何处理受控观察，不能证明真实检索召回率。详情见 examples/README.md。

## 目录与交付材料

| 位置 | 内容 |
| --- | --- |
| src/knowledge_agent/ | 检索工具、LangGraph 决策循环、引用校验、模型配置与 CLI |
| data/ | 10 份合成资料和 20 个验收场景 |
| tests/ | 可重复执行的离线测试 |
| examples/ | 20 组实际输入和输出，19 份 Trace；S19 空问题未调用模型，因此 Trace 为空 |
| docs/01-requirements.md | 需求、范围与成功标准 |
| docs/02-solution.md | 技术方案与模型决策链路 |
| docs/03-evaluation.md | 实际运行结果、人工核对与限制 |
| docs/04-ai-usage.md | AI 辅助范围、用户决策与验证 |
| docs/05-retrospective.md | 难点、学习和下一步 |
| pyproject.toml、uv.lock、.env.example | 依赖与非敏感配置样例 |

自动引用校验只检查文档已读取、段落匹配及引句为原文连续子串；结论是否真正受到引文支持仍需人工判断。模型、工具和提示变更可能导致复跑结果波动，请以实际输出为准。

# 真实运行材料

运行命令：

~~~powershell
uv run python examples/run_live_cases.py --all
~~~

除 S19 外，每个场景都有三份材料：Sxx-input.json 是实际传给 Agent 的问题、历史、模型和控制设置；Sxx-output.json 是程序返回的状态、提案和完整 Trace；Sxx-trace.jsonl 是独立保存的逐步记录。输出中的 trace_path 使用相对仓库根目录的路径，克隆后仍可定位。S19 是空问题，在模型调用前返回 invalid_input，只有输入和输出，Trace 为空。脚本只从 data/scenarios.json 取问题、历史和受控设置，不向模型传入预期答案或验收标准。

| 场景 | 实际状态 | 说明 |
| --- | --- | --- |
| S01 | completed | 真实资料，正常检索、读取及引用核验 |
| S02 | needs_clarification | 真实资料，两份报销制度冲突 |
| S03 | insufficient_evidence | 真实资料，住宿补贴标准缺失 |
| S04 | completed | 年假问题改写后仍找到原文 |
| S05 | completed | 上下文追问转为实习生范围 |
| S06 | failed | 真实模型 + 受控工具持续失败 |
| S07 | limit_reached | 真实模型 + 一步上限 |
| S08 | limit_reached | 真实模型 + 一次工具调用上限 |
| S09 | completed | 真实模型 + 首次搜索人为返回空结果，改写后读取并通过引用核验 |
| S10 | completed | 正式员工出差提前 2 个工作日 |
| S11 | completed | 实习生出差提前 5 个工作日并经导师确认 |
| S12 | needs_clarification | 身份不明时列出不同规则并追问 |
| S13 | completed | V2 格式与明确替代关系 |
| S14 | completed | V2 不支持 WebM |
| S15 | completed | 取消会议室预约提前 2 小时 |
| S16 | completed | 预约会议室提前 1 个工作日 |
| S17 | completed | 实习生年假不适用 |
| S18 | insufficient_evidence | 绩效奖金计算资料缺失 |
| S19 | invalid_input | 空问题在模型与工具调用前拒绝 |
| S20 | completed | 按历史 V1 回答仅支持 MP4 |

S06 和 S09 用于观察真实模型对不同工具返回的后续决策，其工具结果由测试后端控制。completed 只表示机械引文校验通过；语义是否充分仍需人工评估。所有资料均为合成内容。

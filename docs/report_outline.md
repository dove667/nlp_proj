# 长上下文 LLM 课程报告详细大纲

## 文档定位

本文件给出一份可直接落地为课程报告、课程 project final report、或综述型论文初稿的详细大纲。

写作目标：

1. 以 `survey2` 的主线为核心
2. 保留 `survey1`、`survey3` 中可复用的参考材料，但避免重复堆砌
3. 报告整体以“综述为主、实验为支撑”的结构展开
4. 实验部分允许完整展开，不预先人为压缩，最终保留哪些实验可在写作时再裁剪
5. 强调一个清晰的问题意识：
   - 长上下文能力到底主要是 `retrieval` 问题？
   - `reasoning` 问题？
   - `inference-time extension` 问题？
   - 还是 `system serving` 问题？

这份大纲的设计目标是：任何人或任何 AI 在看到它之后，都能理解整篇报告的学术叙事、章节职责、核心论点、实验设计意图与可能的落地写法，并据此直接开始写作。

---

## 一、建议标题

可以从以下几类标题中选择。

### 1. 偏综述型

`Beyond Context Window: A Survey of Long-Context Language Models`

`Long-Context Language Models: Architectures, Evaluation, and Systems`

`长上下文语言模型综述：建模、评测与系统挑战`

### 2. 偏问题驱动型

`What Determines Long-Context Ability in LLMs? A Survey on Retrieval, Reasoning, Inference-Time Extension, and Systems`

`Rethinking Long-Context Ability in LLMs: From Retrieval to Reasoning and Serving`

`长上下文能力究竟是什么：从检索、推理、扩窗到系统的统一视角`

### 3. 偏课程项目型

`A General Survey on Long-Context Language Models with Experimental Perspectives`

`长上下文语言模型综述与实验分析`

如果希望兼顾“综述主导”和“实验存在感”，最推荐：

`What Determines Long-Context Ability in LLMs? A Survey with Experimental Analysis`

---

## 二、摘要应该回答什么

摘要建议覆盖四件事：

1. 问题背景
   - 长上下文能力已成为现代 LLM 的核心能力之一
   - 但“上下文窗口更大”并不等价于“模型更擅长使用长上下文”

2. 本文视角
   - 本文将长上下文能力拆解为 `nominal context length`、`effective retrieval`、`reasoning over long contexts`、`inference-time extension`、`system efficiency` 等维度

3. 综述内容
   - 系统梳理建模方法、上下文扩展方法、检索与记忆增强方法、系统优化技术与评测基准

4. 实验与结论
   - 通过一组实验分析长上下文能力的不同来源与瓶颈
   - 指出长上下文能力并不是单一属性，而是模型、推理策略、任务类型和系统实现共同作用的结果

---

## 三、整篇报告的中心论点

整篇文章最好围绕一个高度集中的核心论点来写：

> 长上下文能力不是一个单一维度，也不能仅由名义上下文窗口长度刻画。
> 它至少由五个彼此关联但可区分的因素共同决定：
> `nominal context length`、`retrieval ability`、`reasoning ability`、`inference-time extension effectiveness`、`system serving efficiency`。

围绕这个中心论点，可以进一步提出三个子命题：

1. 大窗口不等于高有效上下文能力
2. 在很多 benchmark 上，模型可能“找得到信息”，但不一定“用得好信息”
3. 在真实部署中，系统与 serving 问题往往比模型论文中的名义窗口更早成为瓶颈

这三个子命题基本可以串起全文所有章节。

---

## 四、推荐章节结构总览

推荐采用以下主结构：

1. Introduction
2. Problem Formulation and Analysis Framework
3. Technical Foundations of Long-Context Modeling
4. Inference-Time Extension, Retrieval, and Memory Augmentation
5. Systems for Long-Context LLMs
6. Evaluation of Long-Context Ability
7. Experimental Analysis
8. Discussion
9. Conclusion

如果需要更像标准 survey 论文，也可以在 Discussion 前加入：

8. Open Challenges and Future Directions
9. Discussion
10. Conclusion

下面给出建议的详细写法。

---

# 1. Introduction

## 1.1 背景与问题提出

这一节回答：为什么今天要讨论 long-context LLM。

建议内容：

- 大模型的应用正从短问答扩展到长文档阅读、多文档问答、代码仓理解、长对话记忆、智能体规划等任务
- 这些任务都要求模型处理远长于传统几千 token 的上下文
- 工业界和开源社区都在快速推进 32K、128K、1M 甚至更长窗口
- 但越来越多工作表明，名义支持更长上下文，并不意味着模型能稳定完成长距离检索、跨段聚合和复杂推理

## 1.2 问题动机

这一节提出本文真正关注的问题，而不是泛泛地说“长上下文很重要”。

建议直接点题：

> 当一个模型在长上下文任务上失败时，失败究竟来自哪里？
> 是因为它无法从长序列中检索到相关信息？
> 还是因为它能检索但不能推理？
> 抑或是需要依靠推理时扩窗、检索增强或缓存机制才可行？
> 又或者系统层面的延迟、KV cache 与吞吐瓶颈才是真正限制？

## 1.3 本文视角与贡献

这一节给出贡献列表。建议写成 3 到 5 点。

参考写法：

1. 提出一个统一分析框架，将长上下文能力拆解为多个可区分维度
2. 系统综述长上下文 LLM 的主要技术路线，包括架构改造、位置扩展、检索增强、推理时扩展和系统优化
3. 综述主流 benchmark，并分析不同 benchmark 实际测量的是哪一类能力
4. 设计一套实验框架，从 retrieval、reasoning、inference-time extension 和 system serving 的角度分析长上下文能力
5. 总结当前研究的主要共识、争议和未来方向

## 1.4 文章组织

简单说明每章职责，帮助读者建立地图。

---

# 2. Problem Formulation and Analysis Framework

这一章建议作为全文的方法论核心，重要性非常高。

## 2.1 什么是“长上下文能力”

这一节先澄清概念。

建议指出：

- 长上下文能力不是“能接收多少 token”这么简单
- 也不是单个 benchmark 的分数
- 更合理的理解是：模型在长输入条件下保持信息访问、利用、推理和可运行性的综合能力

## 2.2 五维分析框架

建议定义以下五个维度。

### 2.2.1 Nominal Context Length

- 模型或系统声称支持的最大输入长度
- 它通常由位置编码、训练长度、实现限制和推理框架共同决定
- 它是前提条件，但不是充分条件

### 2.2.2 Effective Retrieval Length

- 在给定长度下，模型还能以较高准确率稳定找回目标信息的能力
- 可以通过 NIAH、RULER retrieval 类任务衡量

### 2.2.3 Long-Range Reasoning Ability

- 模型在长输入中完成多跳推理、聚合、对比、跟踪变量和结构化整合的能力
- 这通常比单针检索难得多

### 2.2.4 Inference-Time Extension Effectiveness

- 不改变或少量改变基础模型的情况下，通过推理期技巧提升长上下文表现的能力
- 例如 context extension、sliding window、self-extend、检索路由、chunked reasoning 等

### 2.2.5 System Serving Efficiency

- 系统是否能在长上下文下以可接受的延迟、显存占用和吞吐运行
- 包括 TTFT、prefill latency、decode throughput、KV cache 内存占用、batching 效率等

## 2.3 这五个维度之间的关系

这一节建议给出一张概念图。

要强调：

- `nominal context length` 决定“能不能送进去”
- `retrieval` 决定“能不能找到”
- `reasoning` 决定“能不能用好”
- `inference-time extension` 决定“能不能在不重训的情况下改进”
- `system serving` 决定“能不能真正跑得动”

## 2.4 一个贯穿全文的核心命题

建议把这一章结尾收束到一句话：

> 长上下文能力不是单点指标，而是一个跨模型、跨推理策略、跨任务、跨系统层次的联合性质。

这句话后面可以在结论中再次呼应。

---

# 3. Technical Foundations of Long-Context Modeling

这一章回答：从模型建模角度，研究界如何让 LLM 处理更长上下文。

建议不要简单按年份罗列论文，而是按技术路线组织。

## 3.1 标准 Transformer 的瓶颈

### 3.1.1 全注意力的时间与空间复杂度

- 解释 `O(n^2)` 的来源
- 说明为什么序列长度增长后，问题不仅是 FLOPs，还包括显存与带宽

### 3.1.2 长上下文下的训练与推理瓶颈

- prefill 成本
- decode 阶段 KV cache 膨胀
- HBM 带宽限制

## 3.2 稀疏与局部-全局注意力

可以覆盖：

- Sparse Transformer
- Longformer
- ETC
- BigBird
- Sliding Window Attention

建议讨论：

- 核心思想
- 理论复杂度
- 适合哪些类型任务
- 对复杂长距离推理的局限

## 3.3 线性注意力、低秩近似与核方法

可以覆盖：

- Linformer
- Performer
- Nyströmformer
- Linear Transformers

建议讨论：

- 如何从 `O(n^2)` 走向近线性
- 近似代价是什么
- 为什么很多方法在复杂语言推理上并未完全替代标准注意力

## 3.4 递归、记忆与状态空间路线

可以覆盖：

- Transformer-XL
- Compressive Transformer
- Memorizing Transformer
- S4
- Mamba / Mamba-2 / hybrid variants

建议讨论：

- 为什么这类方法从“显式访问全部历史”转向“压缩状态”
- 它们在超长上下文效率上的优势
- 它们在精确检索、copying、in-context learning 上的挑战

## 3.5 混合架构趋势

可以覆盖：

- Attention + recurrence
- Attention + SSM
- local attention + memory
- hybrid long-context backbones

建议给出观点：

> 近期前沿路线更像是在探索“哪些依赖必须精确注意力，哪些依赖可以压缩/递归化”，而不是完全抛弃 Transformer。

## 3.6 小结

建议总结出这一章的核心结论：

1. 架构层方法解决的是“理论上如何支持更长序列”
2. 但架构扩展不等于长上下文能力必然增强
3. 很多方法提升的是可扩展性，而非所有任务上的有效性

---

# 4. Inference-Time Extension, Retrieval, and Memory Augmentation

这一章回答：如果不完全依赖重训练新架构，现有模型如何在推理时获得更长的可用上下文能力。

## 4.1 位置编码扩展与上下文外推

可以覆盖：

- RoPE scaling
- Position Interpolation
- NTK-aware scaling
- YaRN
- LongRoPE
- LongLoRA

建议重点：

- 这些方法如何把模型“拉长”
- 它们依赖什么假设
- 为什么它们不自动提升复杂 reasoning

## 4.2 推理时扩展策略

可以覆盖：

- Self-Extend
- sliding window decoding
- recurrent chunk processing
- segment-wise inference
- sink tokens / streaming style inference

建议讨论：

- 这类方法不一定改变模型参数
- 它们是在推理调度、上下文重组或缓存机制层面做文章
- 它们很适合课程项目做实验比较

## 4.3 检索增强与层级化上下文组织

可以覆盖：

- RAG
- re-ranking
- chunking strategies
- hierarchical retrieval
- contextual retrieval
- long document routing

建议突出：

- 检索并不是长上下文的替代物，而是重要补充
- 很多真实任务中，“先找再读”比“一次性全塞”更有效率

## 4.4 外部记忆与缓存式增强

可以覆盖：

- external memory
- retrieval memory
- memory bank
- conversational memory
- prefix reuse and caching as functional memory

## 4.5 本章核心结论

建议收束到以下观点：

1. 不改变主模型也可以显著改变长上下文表现
2. inference-time extension 主要改变“如何组织和访问上下文”
3. retrieval 与 memory 方法在真实场景中往往比单纯扩大窗口更具性价比

---

# 5. Systems for Long-Context LLMs

这一章建议单独成章，而且应是本报告的重要特色之一。

它回答：长上下文不只是模型问题，更是系统问题。

## 5.1 为什么系统问题必须单独讨论

建议直接提出：

- 即使模型理论上支持 128K 或 1M，上线时也会遇到 prefill 慢、显存爆炸、吞吐下降、batching 困难等问题
- 因此长上下文能力必须从 serving 视角重新理解

## 5.2 Attention kernel 与 IO-aware optimization

可以覆盖：

- FlashAttention
- FlashAttention-2
- FlashAttention-3

建议讨论：

- IO-aware 的意义
- 为什么长序列瓶颈常常来自内存读写而不是算术运算

## 5.3 KV cache 管理与内存优化

可以覆盖：

- PagedAttention
- KV cache paging
- KV block management
- prefix caching
- prompt caching
- cache reuse

建议讨论：

- 为什么 KV cache 是长上下文推理的核心瓶颈之一
- 为什么缓存命中率会直接影响真实成本

## 5.4 长上下文 serving engine

可以覆盖：

- vLLM
- SGLang
- TensorRT-LLM
- TGI
- DeepSpeed-Inference

建议比较：

- 支持的并行/缓存机制
- 长上下文场景下的优势
- 是否适合研究复现与课程项目

## 5.5 Streaming、Sliding Window 与近似无界上下文

可以覆盖：

- StreamingLLM
- sliding-window serving
- sink tokens
- windowed memory

## 5.6 分布式与大规模 serving

可以覆盖：

- sequence/context parallelism
- prefill-decode disaggregation
- distributed KV cache
- offloading
- memory pooling

## 5.7 这一章应提炼出的系统视角

建议明确几个结论：

1. 真实系统中的长上下文瓶颈通常早于 benchmark 上的理论能力上限
2. 系统优化不会直接提升 reasoning，但会决定实验是否可行、成本是否可接受
3. 同一个模型的长上下文表现，可能会因 serving framework 不同而在延迟和吞吐上出现显著差异

---

# 6. Evaluation of Long-Context Ability

这一章非常关键，因为它决定你如何解释后面的实验。

核心问题是：

> 现有 benchmark 到底在测什么？

## 6.1 为什么单一 benchmark 不够

建议先批判性地指出：

- 单针检索任务过于简单
- 真实长上下文任务包含检索、定位、筛选、聚合、对比和推理多个阶段
- benchmark 的设计会强烈影响“长上下文能力”的定义

## 6.2 Retrieval-oriented benchmarks

### 6.2.1 Needle-in-a-Haystack

介绍：

- 任务形式
- 优点：简单、直观、可扩展长度
- 局限：只能测试基础 retrieval，难以覆盖复杂 reasoning

### 6.2.2 顺序位置与位置鲁棒性测试

可以讨论：

- lost in the middle
- position sensitivity
- placement robustness

## 6.3 Harder synthetic benchmarks

### 6.3.1 RULER

重点介绍：

- 为什么它比 NIAH 更强
- retrieval、aggregation、multi-hop tracing 等任务类别
- effective context length 的意义

### 6.3.2 InfiniteBench、BABILong、NoLiMa、类似任务

可以作为扩展 benchmark 讨论

## 6.4 Real-task long-context benchmarks

### 6.4.1 LongBench

重点介绍：

- 任务类别
- 多文档 QA、摘要、few-shot、代码等设置
- 与合成 benchmark 的区别

### 6.4.2 其他真实任务基准

可以讨论：

- narrative QA
- scientific document QA
- code repository reasoning
- long-dialogue / agent memory tasks

## 6.5 如何建立“benchmark 到能力维度”的映射

这一节是整个评测综述的亮点，建议做成表格。

可以映射为：

- NIAH -> 基础 retrieval
- Lost in the Middle -> 位置鲁棒性
- RULER retrieval -> 多目标检索
- RULER aggregation -> 跨段聚合
- RULER tracing -> 多跳跟踪
- LongBench QA -> 长文理解与问答
- LongBench summarization -> 长输入压缩与信息整合
- serving benchmark -> system efficiency

## 6.6 本章结论

建议强调：

1. benchmark 本身就是理论立场的体现
2. 不能用单一 benchmark 定义长上下文能力
3. 长上下文能力评测必须区分 retrieval、reasoning 与 serving

---

# 7. Experimental Analysis

这一章可以写得完整，最终写作时再决定裁剪哪些实验。

这一章的使命不是“做一个 benchmark leaderboard”，而是服务于全文中心问题：

> 长上下文能力究竟主要受哪些因素决定？

建议实验部分也围绕这个问题组织。

## 7.1 Research Questions

建议明确列出研究问题，例如：

### RQ1

模型在超长输入中的失效首先体现为 retrieval 失败，还是 reasoning 失败？

### RQ2

在相同模型下，推理时扩窗方法能否显著提升有效上下文长度？

### RQ3

不同 benchmark 是否真的在测量相同的长上下文能力？

### RQ4

system serving 指标是否会成为实际长上下文使用的主要限制因素？

### RQ5

名义窗口长度、任务复杂度与系统成本之间是否存在明显错位？

## 7.2 Experimental Setup

### 7.2.1 模型选择

可以列出一个主模型 + 若干对照模型。

例如：

- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct / 1M
- Mistral 系列
- 可选一个更偏长上下文架构或扩窗方法适配更好的模型

### 7.2.2 推理框架

可以包括：

- Hugging Face Transformers
- vLLM
- SGLang

### 7.2.3 长度设置

例如：

- 4K
- 8K
- 16K
- 32K
- 64K
- 128K
- 如资源允许再加入更长长度

### 7.2.4 指标定义

建议至少包含：

- accuracy / exact match
- normalized match
- performance by position
- performance by length
- effective context length
- TTFT
- throughput
- memory usage
- latency breakdown

### 7.2.5 实验控制变量

建议强调：

- 尽量固定模型与 prompt
- 分离模型差异与 serving 差异
- 分离 retrieval 任务与 reasoning 任务

## 7.3 Retrieval-focused experiments

这一节可以对应最基础层。

### 7.3.1 NIAH 基础检索实验

设置建议：

- 不同长度
- 不同 needle 位置
- 多样本重复

分析重点：

- 不同位置是否存在中间衰减
- 随长度增加准确率如何变化

### 7.3.2 多目标检索实验

可以使用：

- RULER 的 multi-key retrieval
- 或自定义多 needle 场景

分析重点：

- 单目标检索强不代表多目标检索强

## 7.4 Reasoning-focused experiments

这一节用于和 retrieval 显式区分。

### 7.4.1 Aggregation tasks

例如：

- common words
- frequency counting
- cross-segment evidence aggregation

### 7.4.2 Multi-hop tracing tasks

例如：

- variable tracking
- chain tracing
- multi-document clue composition

### 7.4.3 真实任务中的 reasoning

可选：

- LongBench QA 子集
- Qasper
- HotpotQA / 2WikiMultihopQA / MuSiQue

分析重点：

- retrieval 得分高时，reasoning 是否仍然下降
- reasoning 任务是否比 retrieval 更快出现随长度退化

## 7.5 Inference-time extension experiments

这一节非常适合体现“不是只有模型本身重要”。

### 7.5.1 不同扩窗/推理时策略比较

可考虑：

- baseline direct long-context inference
- rope scaling / yarn / interpolation variant
- self-extend
- sliding-window + memory style inference
- retrieval-assisted inference

### 7.5.2 对不同任务类型的影响

分析重点：

- 扩窗方法对 retrieval 帮助多大
- 对 reasoning 帮助是否有限
- 对真实任务帮助是否依赖任务结构

### 7.5.3 有效上下文长度提升分析

建议输出：

- 不同方法下的 effective context length
- performance decay slope

## 7.6 System serving experiments

这一节建议作为实验大章中的重点之一。

### 7.6.1 单模型在不同长度下的 serving 成本

指标：

- TTFT
- prefill latency
- decode speed
- peak memory
- tokens/s

### 7.6.2 不同 serving engine 对比

例如：

- Transformers baseline
- vLLM
- SGLang

比较点：

- 长上下文下的吞吐
- 缓存管理效率
- batch 扩展能力
- 稳定性与易用性

### 7.6.3 Prefix caching / cache reuse 分析

如果实验条件允许，可以分析：

- 重复前缀时的 latency 变化
- 长 system prompt / repeated context 下的加速效果

### 7.6.4 批处理与并发实验

可以测试：

- batch size 对吞吐的影响
- 长上下文对并发能力的压制程度

### 7.6.5 System cost vs task accuracy

这一节非常有价值。

建议问：

- 更高 system cost 是否带来更好任务效果？
- 哪些情况下“花更多代价读更长上下文”其实并不划算？

## 7.7 Unified Analysis

这一节把前面实验收束成结论。

建议围绕以下问题展开：

1. retrieval 与 reasoning 哪个更早失效
2. inference-time extension 主要改善了什么，没改善什么
3. system 优化主要改变了什么，没改变什么
4. 名义窗口、有效窗口、系统可运行窗口之间是否存在系统性差距

## 7.8 Threats to Validity

建议加入，显得更完整。

包括：

- benchmark 选择有限
- 模型数量有限
- 结果依赖 prompt 和 decoding 设置
- 开源 serving 框架配置差异会影响结论
- 课程项目资源下无法覆盖所有超大规模设置

---

# 8. Discussion

这一章非常重要，用来从“结果与综述内容”中提炼更高层理解。

## 8.1 长上下文能力首先是什么问题

建议讨论：

- 它不是单纯 retrieval
- 也不是纯 reasoning
- 它是“信息访问 + 信息整合 + 可运行性”的复合问题

## 8.2 名义窗口与有效窗口的错位

建议强调：

- 模型标称支持 128K/1M，并不意味着在这些长度上仍有稳定表现
- effective context length 应成为比 nominal window 更重要的概念

## 8.3 为什么 retrieval benchmark 不足够

建议讨论：

- 单针检索不能代表多跳与聚合
- 真实应用更接近结构化筛选和多证据整合

## 8.4 inference-time extension 的价值与边界

建议讨论：

- 它是高性价比手段
- 但常常更像“让信息可被访问”，未必根治 reasoning 短板

## 8.5 system 视角带来的重新理解

建议强调：

- 对真实应用来说，是否可服务、可部署、可缓存，往往和模型精度同样重要
- 因此 long-context research 不应只停留在模型结构层面

---

# 9. Open Challenges and Future Directions

如果篇幅允许，建议单独成章。

## 9.1 更合理的能力定义

- 如何统一 nominal context、effective context 与 utility
- 如何定义“真正有用的长上下文能力”

## 9.2 更强的 reasoning benchmark

- 需要比 NIAH 更复杂的长上下文推理测试
- 需要区分 retrieval success 与 reasoning success

## 9.3 Retrieval、reasoning 与 memory 的统一建模

- 未来研究应关注“何时检索、何时缓存、何时全读”

## 9.4 模型与系统协同设计

- architecture-system co-design
- KV-aware model design
- serving-aware training

## 9.5 长上下文在真实应用中的新场景

- agents
- codebase understanding
- scientific literature synthesis
- legal / medical document intelligence

---

# 10. Conclusion

结论建议收束到三层：

## 10.1 总结全文

指出本文系统回顾了：

- 长上下文建模路线
- 推理时扩展方法
- 检索与记忆增强
- 系统 serving 技术
- benchmark 与实验分析

## 10.2 最重要的最终结论

建议明确写出：

> 长上下文能力不能由上下文窗口大小单独定义。
> 一个模型是否真正具备长上下文能力，取决于它在长输入下的检索、推理、推理期扩展适配能力以及系统可运行性。

## 10.3 对课程项目的落点

建议最后落在：

- 这不仅是一个 survey 问题，也是一个实验与系统问题
- 理解这些维度的差异，有助于更合理地设计未来的长上下文模型与应用系统

---

## 五、推荐图表与表格清单

为了让这篇报告更像“综述论文级别”的成品，建议至少准备以下图表。

## 5.1 概念图

### 图 1：长上下文能力五维框架图

展示：

- nominal length
- retrieval
- reasoning
- inference-time extension
- system efficiency

### 图 2：技术路线全景图

展示：

- 架构改造
- 位置扩展
- retrieval / memory
- systems
- benchmark

### 图 3：benchmark 到能力维度的映射图

## 5.2 时间线

### 图 4：长上下文研究发展时间线

可以复用 `survey` 中已有时间线材料，但要统一风格，避免三份 survey 的重复内容直接拼接。

## 5.3 表格

### 表 1：长上下文方法分类总表

列建议：

- 方法类别
- 代表工作
- 核心思想
- 复杂度
- 优势
- 局限

### 表 2：长上下文 benchmark 对比表

列建议：

- benchmark
- 任务类型
- 测量能力
- 是否 synthetic
- 是否强调 reasoning
- 主要局限

### 表 3：system serving 技术总表

列建议：

- 技术 / 系统
- 核心优化对象
- 解决的问题
- 对长上下文的意义

### 表 4：实验矩阵总表

列建议：

- 模型
- benchmark
- 长度
- 方法
- system backend
- 指标

## 5.4 实验图

建议输出：

- NIAH heatmap
- RULER accuracy vs length
- different task categories comparison
- latency / throughput vs context length
- memory usage vs context length
- effective context length comparison

---

## 六、建议的写作顺序

为了提高实际产出效率，推荐不要按章节顺序硬写，而是按以下顺序推进。

## 6.1 第一步：先固定核心框架

先写：

- Introduction
- Section 2 分析框架

因为这两部分决定全文叙事。

## 6.2 第二步：整合三份 survey 的可复用内容

以 `survey2` 为主，吸收 `survey1` 和 `survey3` 中：

- 时间线
- 方法分类
- benchmark 讨论
- system 讨论

但一定要避免重复堆砌。

## 6.3 第三步：补 Evaluation 和 Systems 两章

这两章很可能是这篇报告的辨识度来源。

## 6.4 第四步：最后写实验章

因为实验章应该为前面理论服务，而不是反过来决定全文结构。

## 6.5 第五步：最后回写摘要与结论

摘要必须在全文定稿后再写，否则很容易泛泛而谈。

---

## 七、如果后续要继续让我协作，推荐的下一步

在这份大纲基础上，后续最自然的协作顺序是：

1. 把这份大纲扩成“目录 + 每节写作要点”
2. 先统一 `survey1/2/3` 的素材分工
3. 直接起草 `Introduction` 和 `Section 2`
4. 再生成一版实验章节模板
5. 最后补图表与参考文献框架

如果继续推进，最推荐的下一步是：

`先把 survey2 升级成按本大纲重组的正式初稿结构。`

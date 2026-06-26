# PruneRAG Speaker Notes

## English Version

### Slide 1: Title
Good morning everyone. Today I will present our work, PruneRAG: Confidence-Guided Query Decomposition Trees for Efficient Retrieval-Augmented Generation. This work focuses on a central challenge in complex question answering: how to retrieve the right evidence, reason over it reliably, and avoid unnecessary retrieval cost. I will show that by organizing reasoning as a query decomposition tree and using confidence to guide pruning, PruneRAG can improve both effectiveness and efficiency.

### Slide 2: Outline
The talk has four parts. First, I will introduce the background of retrieval-augmented generation and the motivation behind this work. Second, I will describe the PruneRAG framework, including adaptive node expansion, confidence-guided pruning, and fine-grained entity retrieval. Third, I will present experimental results and analysis. Finally, I will summarize the key takeaways and discuss future directions.

### Slide 3: Background
Retrieval-augmented generation, or RAG, has become a widely used paradigm for knowledge-intensive question answering. Instead of relying only on the parametric knowledge of a language model, RAG retrieves external documents during inference and uses them as evidence. However, multi-hop questions are more difficult, because they require connecting evidence across multiple documents. Existing methods often use either chain-based reasoning or tree-based exploration, but they still lack a principled mechanism to decide when to stop, when to split a query, and when to refine retrieval.

### Slide 4: Evidence Forgetting
The first problem we study is evidence forgetting. In many cases, a model has already retrieved the key evidence, but it fails to use that evidence correctly in later reasoning. This can happen because early evidence is diluted by accumulated context, or because errors propagate from intermediate steps. As shown here, mainstream baselines can have high evidence forgetting rates. PruneRAG addresses this by rejecting low-confidence answers and forcing the system to reflect, decompose, or retrieve more precisely before committing to a final answer.

### Slide 5: Inefficiency
The second problem is inefficiency. Tree-based RAG methods can explore many branches, but not all branches are useful. Without expansion control, they may generate low-value nodes, repeat retrieval calls, and increase latency. For example, RAG-Star has high retrieval cost and long inference time. The key missing component is a confidence-guided control mechanism that can accept reliable answers early and avoid unnecessary exploration.

### Slide 6: PruneRAG Overview
This slide shows the overall framework of PruneRAG. The system builds a query decomposition tree from the original complex question. Each node is processed adaptively: it may be answered directly, decomposed into sub-queries, or converted into entity-based retrieval. Then confidence-guided pruning decides whether a generated answer is reliable enough to keep. Finally, fine-grained retrieval uses entity anchors when ordinary query decomposition is no longer effective.

### Slide 7: Tree Structure
PruneRAG uses two types of nodes. A query node contains a sub-query, retrieved documents, and a candidate answer. An entity node contains salient entities and the documents retrieved through them. Entity nodes are terminal nodes, used when the query cannot be answered or decomposed further. This tree structure allows PruneRAG to combine top-down decomposition with bottom-up answer aggregation.

### Slide 8: Adaptive Node Expansion
For each node, PruneRAG makes a three-way decision. If the current evidence is sufficient, the model generates an answer and stops expanding that node. If the query is not answerable but can be decomposed, it is split into two sub-queries. If it is neither answerable nor decomposable, the system extracts key entities and performs entity-level retrieval. This adaptive strategy controls both the width and depth of the reasoning tree.

### Slide 9: Confidence-Guided Pruning
The core pruning signal is answer confidence, computed from token-level log probabilities. If the confidence of a candidate answer is above the threshold, the answer is accepted as a reliable leaf. If the confidence is below the threshold, the answer is rejected and the system continues to decompose or retrieve. This mechanism reduces error propagation, because uncertain answers are not allowed to contaminate later reasoning.

### Slide 10: Fine-Grained Entity Retrieval
When a query cannot be decomposed further, PruneRAG switches to fine-grained entity retrieval. The system treats the query as a minimal semantic unit, extracts important entities, and uses them as structured retrieval anchors. This improves retrieval precision and reduces noise. In practice, entity retrieval is especially useful for difficult questions where the next useful evidence depends on a specific person, location, event, or relation.

### Slide 11: Bottom-Up Backtracing
After the tree is constructed, PruneRAG performs bottom-up backtracing. Answer leaves return their accepted answers. Entity leaves summarize the evidence retrieved through entity anchors. Internal nodes aggregate the results of their children. This process produces the final answer at the root. The case study shows how PruneRAG avoids evidence forgetting by pruning unreliable branches and recovering a more coherent reasoning path.

### Slide 12: Algorithm
The inference procedure has two stages. In the top-down stage, PruneRAG starts from the original question, retrieves documents, generates a candidate answer, computes confidence, and decides whether to accept, decompose, or use entity retrieval. In the bottom-up stage, the system aggregates information from leaves to root. The important point is that pruning and expansion are not separate heuristics; they work together to make the tree both reliable and efficient.

### Slide 13: Evidence Forgetting Rate
To evaluate evidence misuse, we propose Evidence Forgetting Rate, or EFR. EFR measures the proportion of cases where all golden evidence has already been retrieved, but the model still predicts a wrong answer. This metric separates retrieval failure from reasoning failure. A high EFR means the system did find the necessary evidence, but failed to use it correctly. Therefore, EFR directly captures the forgetting problem we aim to solve.

### Slide 14: Experimental Setup
We evaluate PruneRAG on five main QA datasets, including HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle, and GPQA. These datasets cover different types of multi-hop and complex reasoning. We test Llama and Qwen backbones at different scales, and compare against both chain-based and tree-based baselines. In the main setup, we use BGE-large-en-v1.5 over the 2018 Wikipedia dump, retrieve the top five documents for each sub-query, and set the branching factor to two, maximum depth to three, and confidence threshold to 0.95.

### Slide 15: Main Results
Here are the main results using Qwen3-8B. PruneRAG achieves the best performance across the three multi-hop QA datasets shown here. It improves exact match and F1 while also reducing evidence forgetting. More importantly, it is much faster than multi-retrieval baselines because it avoids unnecessary exploration. This shows that confidence-guided pruning does not simply improve accuracy; it also makes inference more efficient.

### Slide 16: Ablation Study
The ablation study helps us understand which components matter most. Removing confidence increases EFR, especially on MuSiQue, showing that unreliable answers can disrupt evidence transmission. Removing the answer branch prevents early termination and increases both retrieval calls and forgetting. Removing adaptive expansion causes strong degradation across most datasets because the tree grows without proper control. These results confirm that confidence pruning, answer-based early stopping, entity retrieval, and adaptive expansion all contribute to the final performance.

### Slide 17: Retrieval Efficiency
This slide analyzes retrieval efficiency. We compare retrieval number, recall, and retrieval efficiency. PruneRAG uses fewer retrieval calls while maintaining strong recall and answer accuracy. In the bubble chart, PruneRAG appears in the desirable region: low retrieval count and large EM, with competitive recall. This suggests that PruneRAG retrieves more selectively and uses the retrieved evidence more effectively, rather than simply retrieving more documents.

### Slide 18: Hyperparameter Sensitivity
We also study the sensitivity of key hyperparameters. For maximum tree depth, multi-hop datasets usually benefit from depth two or three, while single-hop questions prefer shallower trees. For the confidence threshold, performance is stable around 0.9 to 0.95. This indicates that PruneRAG is reasonably robust and does not require heavy tuning for each dataset.

### Slide 19: Larger Models and Generalization
We further test PruneRAG with larger backbone models. The results show that scaling the model alone does not fully solve evidence forgetting. Even larger models can still retrieve the right evidence but fail to use it correctly. PruneRAG continues to improve performance and reduce EFR with larger models and alternative retrievers, suggesting that the problem is not tied to a specific model or retrieval backend.

### Slide 20: Conclusion
To conclude, PruneRAG introduces a confidence-guided query decomposition tree for efficient and reliable RAG. It improves structured reasoning by combining answer generation, decomposition, and entity retrieval in one tree. It reduces evidence forgetting through confidence-guided pruning. It also improves efficiency by avoiding unnecessary retrieval and expansion. Overall, PruneRAG provides a practical framework for complex question answering where both accuracy and cost matter. Thank you, and I am happy to take questions.

## 中文翻译

### 第 1 页：标题
大家早上好。今天我将介绍我们的工作 PruneRAG：一种基于置信度引导的查询分解树，用于高效的检索增强生成。这个工作关注复杂问答中的一个核心问题：如何检索到正确证据，可靠地进行推理，并避免不必要的检索开销。我会展示 PruneRAG 如何通过查询分解树组织推理，并利用置信度进行剪枝，从而同时提升效果和效率。

### 第 2 页：大纲
本次报告分为四个部分。首先，我会介绍检索增强生成的背景以及本工作的研究动机。第二，我会说明 PruneRAG 的整体框架，包括自适应节点扩展、置信度引导剪枝和细粒度实体检索。第三，我会展示实验结果与分析。最后，我会总结主要结论，并讨论未来可能的研究方向。

### 第 3 页：背景
检索增强生成，也就是 RAG，已经成为知识密集型问答中的常用范式。它不只依赖语言模型的参数知识，而是在推理时检索外部文档，并将其作为证据。然而，多跳问题更加困难，因为它需要跨多个文档连接证据。现有方法通常采用链式推理或树式探索，但它们仍然缺少一个有原则的机制来决定何时停止、何时拆分问题，以及何时细化检索。

### 第 4 页：证据遗忘
我们研究的第一个问题是证据遗忘。在很多情况下，模型其实已经检索到了关键证据，但在后续推理中没有正确利用这些证据。原因可能是早期证据被不断累积的上下文稀释，也可能是中间步骤的错误继续传播。这里可以看到，主流基线方法存在较高的证据遗忘率。PruneRAG 通过拒绝低置信度答案，并迫使系统继续反思、分解或进行更精确的检索，从而避免过早提交错误答案。

### 第 5 页：低效率
第二个问题是低效率。树式 RAG 方法可以探索多个分支，但并不是所有分支都有价值。如果没有扩展控制，它们可能生成大量低价值节点，重复检索相同信息，并增加推理延迟。例如，RAG-Star 的检索成本和推理时间都较高。这里缺少的关键组件，是一个由置信度引导的控制机制，用于提前接受可靠答案，并避免不必要的探索。

### 第 6 页：PruneRAG 总览
这一页展示了 PruneRAG 的整体框架。系统从原始复杂问题出发，构建一棵查询分解树。每个节点都会被自适应处理：它可能被直接回答，也可能被分解为子问题，或者转化为基于实体的检索。随后，置信度引导剪枝会判断生成答案是否足够可靠。最后，当普通查询分解不再有效时，细粒度实体检索会使用实体锚点来提高检索精度。

### 第 7 页：树结构
PruneRAG 使用两类节点。查询节点包含子查询、检索到的文档和候选答案。实体节点包含显著实体以及通过这些实体检索到的文档。实体节点是终止节点，用于处理查询无法继续回答或分解的情况。这种树结构使 PruneRAG 能够结合自顶向下的问题分解和自底向上的答案聚合。

### 第 8 页：自适应节点扩展
对于每个节点，PruneRAG 会做出三种决策。如果当前证据足够，模型就生成答案，并停止扩展该节点。如果查询无法直接回答但可以分解，就将其拆成两个子查询。如果它既无法回答也无法分解，系统就抽取关键实体，并进行实体级检索。这种自适应策略能够同时控制推理树的宽度和深度。

### 第 9 页：置信度引导剪枝
核心剪枝信号是答案置信度，它由 token 级别的对数概率计算得到。如果候选答案的置信度高于阈值，该答案就会被接受为可靠叶节点。如果置信度低于阈值，该答案会被拒绝，系统继续进行分解或检索。这个机制能够减少错误传播，因为不确定的答案不会继续污染后续推理。

### 第 10 页：细粒度实体检索
当一个查询无法进一步分解时，PruneRAG 会切换到细粒度实体检索。系统将该查询视为最小语义单元，抽取重要实体，并将它们作为结构化检索锚点。这可以提高检索精度并减少噪声。在实际场景中，当下一个有用证据依赖于特定人物、地点、事件或关系时，实体检索尤其有帮助。

### 第 11 页：自底向上回溯
树构建完成后，PruneRAG 会执行自底向上的回溯。答案叶节点返回已接受的答案。实体叶节点总结通过实体锚点检索到的证据。内部节点聚合其子节点结果。最终，根节点输出最终答案。这个案例展示了 PruneRAG 如何通过剪除不可靠分支，恢复更加连贯的推理路径，从而避免证据遗忘。

### 第 12 页：算法流程
推理过程包括两个阶段。在自顶向下阶段，PruneRAG 从原始问题开始，检索文档，生成候选答案，计算置信度，并决定接受、分解，还是使用实体检索。在自底向上阶段，系统从叶节点到根节点聚合信息。关键点在于，剪枝和扩展不是两个孤立的启发式操作，而是共同作用，使整棵树既可靠又高效。

### 第 13 页：证据遗忘率
为了评估证据误用问题，我们提出了证据遗忘率 EFR。EFR 衡量的是这样一类样本的比例：所有黄金证据都已经被检索到，但模型仍然给出了错误答案。这个指标将检索失败和推理失败区分开来。高 EFR 说明系统已经找到了必要证据，却没有正确使用它。因此，EFR 能直接刻画我们希望解决的证据遗忘问题。

### 第 14 页：实验设置
我们在五个主要问答数据集上评估 PruneRAG，包括 HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle 和 GPQA。这些数据集覆盖了不同类型的多跳和复杂推理问题。我们测试了不同规模的 Llama 和 Qwen 骨干模型，并与链式方法和树式方法进行比较。在主要实验设置中，我们使用 BGE-large-en-v1.5 检索器和 2018 Wikipedia 语料库，每个子查询检索 top-5 文档，并将最大分支因子设为 2、最大树深度设为 3、置信度阈值设为 0.95。

### 第 15 页：主要结果
这里展示的是使用 Qwen3-8B 的主要结果。PruneRAG 在这三个多跳问答数据集上都取得了最佳表现。它提升了 EM 和 F1，同时降低了证据遗忘率。更重要的是，由于避免了不必要的探索，它比多轮检索基线快很多。这说明置信度引导剪枝不仅提升准确率，也提升了推理效率。

### 第 16 页：消融实验
消融实验帮助我们理解各个组件的作用。去掉置信度会提高 EFR，尤其是在 MuSiQue 上，说明不可靠答案会干扰关键证据传递。去掉答案分支会阻止早停，导致检索次数和遗忘率都上升。去掉自适应扩展会在多数数据集上造成明显退化，因为推理树会失去有效控制。这些结果说明，置信度剪枝、基于答案的早停、实体检索和自适应扩展都对最终性能有贡献。

### 第 17 页：检索效率
这一页分析检索效率。我们比较了检索次数、召回率和检索效率。PruneRAG 使用更少的检索调用，同时保持较强的召回率和答案准确率。在气泡图中，PruneRAG 位于比较理想的区域：检索次数低、EM 较高，并且召回率具有竞争力。这说明 PruneRAG 不是简单地检索更多，而是更有选择性地检索，并且更有效地利用已检索到的证据。

### 第 18 页：超参数敏感性
我们还研究了关键超参数的敏感性。对于最大树深度，多跳数据集通常在深度 2 或 3 时效果较好，而单跳问题更适合浅层树。对于置信度阈值，性能在 0.9 到 0.95 附近比较稳定。这表明 PruneRAG 具有较好的鲁棒性，不需要为每个数据集进行大量调参。

### 第 19 页：大模型与泛化性
我们进一步在更大的骨干模型上测试 PruneRAG。结果表明，仅仅扩大模型规模并不能完全解决证据遗忘问题。即使是更大的模型，也可能检索到正确证据却没有正确利用。PruneRAG 在大模型和不同检索器下仍然能够提升性能并降低 EFR，这说明该问题并不依赖于某个特定模型或检索后端。

### 第 20 页：总结
最后总结一下，PruneRAG 提出了一种置信度引导的查询分解树，用于实现高效且可靠的 RAG。它通过在同一棵树中结合答案生成、问题分解和实体检索，提升了结构化推理能力。它通过置信度引导剪枝减少证据遗忘。它还通过避免不必要的检索和扩展提升了效率。总体而言，PruneRAG 为复杂问答提供了一个兼顾准确率和成本的实用框架。谢谢大家，欢迎提问。

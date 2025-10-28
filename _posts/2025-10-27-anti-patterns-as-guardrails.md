---
title: "Exploring Continuous Learning: Reasoning Bank + Recursive Language Models"
date: 2025-10-27
categories: [Memory, Reason]
tags: [llm, memory, reasoning, pubmedqa]
math: true
toc: true
render_with_liquid: false
---

## TLDR

A tiny [Reasoning Bank](https://arxiv.org/html/2509.25140v1) of success and failure reasoning strategies on PubMedQA nudged accuracy from 73% → 77%. The lift was strongest when both success and failure strategies were present, reaching 84% in that subset. No fine‑tuning, just retrieval of distilled memories.

---

## Motivation

2025 gave us million token context windows, and a new failure mode in [context rot](https://research.trychroma.com/context-rot). Retrieval is as important as ever (I'm avoiding RAG here, as I agree with Jeff Huber's point that we should just call it retrieval), and I'm interested in exploring memory and reasoning, vs just stuffing prompts with relevant chunks.

So as of late Oct 2025, expanding the context window of LLMs, and in some ways continual learning for LLMs, is a trending topic. I'm still reading and learning, and I'd bucket recent papers I've come across (thanks to https://www.alphaxiv.org/) into three categories (these are my working buckets as I learn and build more, not meant to be an academic taxonomy):

1. architectural changes - eg [Native Sparse Attention](https://arxiv.org/abs/2502.11089) for long-horizon efficiency 
2. parametric updates - eg online SFT/RL such as [Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103), which allows continuous knowledge updates without impacting generalization
3. "outer loop" engineering - building architecture (or scaffold?) around the llm, eg [LightMem](https://www.arxiv.org/abs/2510.18866), [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/), [Reasoning Bank](https://arxiv.org/html/2509.25140v1), and many others

I was especially interested in recursive language model, and reasoning bank, because it seems to me that they align well with the [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html): 

_"One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are *search and learning*."_

In the case of Recursive Language Models, we give the AI Agent full autonomy to *search* in whatever way it deems. We are not crafting tools with specific instructions on how to use them, which [clog up context](https://www.reddit.com/r/Anthropic/comments/1nkdtiw/mcp_server_context_rot/). Instead, allowing the LLM an execution environment to create whatever python code it deems necessary, and spawn up additional LLM processes to gather the required information.

For Reasoning Bank, we are letting the LLM reflect and *learn* from its own reasoning traces on both success, and more importantly, the incorrect predictions. Unlike many parameter-update methods which often depend on preference winners or scalar rewards, a memory layer allows the LLM to reflect when it got something wrong, and try to distill the "why".

For this post, we will focus on the PubMedQA dataset, which has fairly limited context windows, so naturally implementing Recursion did not help with performance. We'll save the combo of RLM+RB for another post with experiments on [BrowseComp Plus dataset](https://arxiv.org/abs/2508.06600).

---

## Core idea around Reasoning Bank

When humans learn complex tasks, we benefit from two types of examples:
1. **Success patterns**: "Here's how to solve this correctly"
2. **Failure patterns**: "I got this wrong, here's what I should watch out for next time"

Most AI systems only learn from successes. But what if the LLM can learn from both success and failures? And perhaps even more importantly, reflect on why it made a mistake in the first place, so the learnings are generalizable?

---

## Experiment

- **Dataset**: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
- **Task**: Answer yes/no/maybe based on research abstracts
- **LLM**: Gemini 2.5 Flash
- **Memory Reasoning Bank**: 200 training examples → 144 success patterns + 56 failure-patterns
- **Retrieval of Memory**: Semantic similarity matching via embeddings
- **Train/Test**: Randomly sample, build ("train") the Reasoning Bank on 200 examples, inference on 100 examples
- **Comparison**: Vanilla LLM prediction: 73% accuracy, LLM+MRB: 77% accuracy. Across 10 independent random splits/seeds, the memory‑augmented setup delivered a consistent ~4‑point lift; small but stable, not a one‑off.

---

## Learning from Failure

An failure-pattern is a structured analysis of a reasoning and prediction failure

### Extraction

When the LLM produces an incorrect answer, a LLM Judge extracts an failure-pattern with these components:

1. **Error Identification**: What specific reasoning error occurred?
2. **Warning Signals**: What indicators should have been noticed?
3. **Correct Approach**: How should the reasoning have proceeded?
4. **Generalizable Lessons**: What broader takeaways apply to similar questions?

**LLM Judge Prompt**:

```plaintext
Your task: analyze this failed reasoning trace and extract one failure-pattern
(common reasoning error) as a cautionary example.

Question: {query}
Reasoning trace: {reasoning}
Outcome: failure
Ground truth: {correct_answer}
Model answer: {wrong_answer}

Extract an failure-pattern with:
1. Title: Short name for the error (e.g., "Overgeneralization from Limited Evidence")
2. Description: One-sentence summary of when this error occurs
3. Content: 4-component analysis:
   - ERROR: What reasoning mistake was made
   - WARNING SIGNALS: What should have raised concerns
   - CORRECT APPROACH: How to reason properly
   - LESSONS: Generalizable takeaways
4. Tags: Domain/task filters (["pubmedqa", "yes_no_qa", ...])
5. Outcome: "failure"
```

**Example failure-pattern Output**:

JSON Structure:

```json
{
  "title": "Overgeneralization from Limited Evidence",
  "description": "Use as warning when tendency to dismiss feasibility based solely on preliminary/limited data",
  "content": "...",
  "tags": ["pubmedqa", "yes_no_qa", "feasibility_study"],
  "outcome": "failure"
}
```

Content Field (formatted):

```plaintext
ERROR: Dismissed feasibility of intervention based on limited pilot evidence,
failing to recognize that feasibility studies are designed to test practicality,
not efficacy.

WARNING SIGNALS:
- Question asks about feasibility, not efficacy
- Study explicitly states "feasibility study"
- Evidence shows completion rates and participant feedback

CORRECT APPROACH:
1. Distinguish feasibility questions from efficacy questions
2. Recognize that feasibility focuses on practical implementation
3. Evaluate completion rates and participant acceptance
4. Answer 'yes' for feasibility even if efficacy unclear

LESSONS: Don't apply efficacy standards to feasibility questions.
Limited evidence can still demonstrate feasibility.
```

---

### Injecting Reasoning Memories

**Purpose**: Inject **both** success patterns AND failure-patterns with distinct formatting

**Code Implementation**:

```python
# Inject success pattern (if retrieved)
if strategies and len(strategies) > 0:
    prompt += "\n\n" + "="*60 + "\n"
    prompt += "RELEVANT STRATEGY (from similar past questions):\n"
    prompt += "="*60 + "\n\n"
    strategy = strategies[0]
    prompt += f"{strategy.title}\n"
    prompt += f"When to use: {strategy.description}\n"
    prompt += f"Approach:\n{strategy.content}\n\n"

# Inject failure-pattern (if retrieved) with warning formatting
if anti_patterns and len(anti_patterns) > 0:
    prompt += "="*60 + "\n"
    prompt += "⚠️  ANTI-PATTERN TO AVOID:\n"
    prompt += "="*60 + "\n\n"
    anti = anti_patterns[0]
    prompt += f"{anti.title}\n"
    prompt += f"Common mistake: {anti.description}\n"
    prompt += f"Analysis:\n{anti.content}\n\n"

# Closing guidance
if strategies or anti_patterns:
    prompt += "="*60 + "\n"
    prompt += "Consider these patterns when answering.\n"
    prompt += "="*60 + "\n\n"
```

**Prompt with injected Reasoning Memories**:

```plaintext
Based on the provided medical research context, answer the following
question with 'yes', 'no', or 'maybe':

Question: Aromatase inhibitor-related musculoskeletal symptoms: is preventing
them with exercise feasible?

Context:
[... medical abstract about feasibility study ...]

============================================================
RELEVANT STRATEGY (from similar past questions):
============================================================

Evaluate Feasibility Study Outcomes
When to use: Use when assessing practical implementation feasibility
Approach:
1. Check completion rates and participant retention
2. Evaluate acceptability metrics
3. Distinguish feasibility from efficacy
4. Answer based on implementation success

============================================================
ANTI-PATTERN TO AVOID
============================================================

Overgeneralization from Limited Evidence
Common mistake: Dismissing feasibility based on limited pilot data

Analysis:
ERROR: Dismissed feasibility of intervention based on limited pilot
evidence, failing to recognize that feasibility studies test practicality,
not efficacy.

WARNING SIGNALS:
- Question asks about feasibility, not efficacy
- Study explicitly states "feasibility study"
- Evidence shows completion rates and participant feedback

CORRECT APPROACH:
1. Distinguish feasibility questions from efficacy questions
2. Recognize that feasibility focuses on practical implementation
3. Evaluate completion rates and participant acceptance
4. Answer 'yes' for feasibility even if efficacy unclear

LESSONS: Don't apply efficacy standards to feasibility questions.
Limited evidence can still demonstrate feasibility.

============================================================
Consider these patterns when answering.
============================================================

Please analyze the context carefully and provide your answer as one
word: yes, no, or maybe.
```

This structure allows flexibility on the presence or absence of memories. One important finding is that ~19% of examples had both a successful and a failure pattern extracted, and those yielded 84% accuracy (73% baseline for vanilla LLM predictions).

---

### Inference flow


```plaintext
Input
  │
Retriever ──► Top‑k Memories
  │              ├─ Strategy: "Evaluate Feasibility (completion, retention...)"
  │              └─ Anti‑pattern: "Overgeneralize from limited evidence"
  │
Prompt Composer
  │   (inject strategy/anti‑pattern blocks)
  ▼
LLM ⇢ answer (yes/no/maybe)
  │
Judge (optional) ──► new memory (strategy or anti‑pattern)
```

---

## Limitations & Next Steps

This is 1 of N posts on continual learning, I just wanted to document learnings so far.

### Current Limitations

1. only tested on PubMedQA for now 
2. while everything else about the experiment is very much aligned with bitter lesson's thesis of leaning into search and learning, there is still a "hand crafted" parameter of embedding similarity at the retrieval step

### What's Next

1. the reasoning bank was built ("trained", but not in the classical ML sense) on ~200 examples, what if we scaled it to more?
2. as successful and failure patterns increased, is there a way to "consolidate" ([LightMem](https://arxiv.org/abs/2510.18866?utm_source=chatgpt.com)) the memories so that the reasoning traces and failure reflections are even more generalizable and helpful?
3. can the "training" step of building up the reasoning bank also be more iterative, in that example 10 references reason traces developed from examples 1-9, and potentially even updates the memories given some sort of ground truth label?
4. I'm curious how this approach compares to using [GEPA](https://arxiv.org/abs/2507.19457). I intentionally did not tune the prompt (thanks Claude) for this current experiment
5. very curious on trying the combo of RLM + MRB on BrowserComp Plus, as reasoning memories on how the LLM recursively found solutions to complex and long context could be incredibly valuable

---

*This work was inspired by [ReasoningBank](https://arxiv.org/html/2509.25140v1), [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) and builds on the [PubMedQA dataset](https://pubmedqa.github.io/).*

*Views are strictly my own. Experiments based only on public datasets.*

*Published: October 27, 2025*
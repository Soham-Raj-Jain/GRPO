# README - Colab 4: GRPO Reasoning Model Training

### Overview

This notebook demonstrates Group Relative Policy Optimization (GRPO) for training reasoning models. GRPO teaches models to solve problems by generating multiple solutions, evaluating correctness, and learning from successful reasoning patterns.

### What You'll Learn

- How GRPO differs from supervised learning
- Training models to reason step-by-step
- Creating reward functions for verification
- Self-supervised learning from model outputs
- Building reasoning capabilities
- Similar techniques to OpenAI's o1 model

### Requirements

- Google Colab with T4 GPU (free tier)
- Approximately 20-25 minutes of training time
- Problems with verifiable correct answers

### How GRPO Works

1. Given a problem, generate N solutions (N=4-8)
2. Evaluate each solution with reward function
3. Calculate group statistics (mean reward)
4. Update model to favor high-reward solutions
5. Learn from self-generated successful reasoning

### Dataset Format

Problems with verifiable answers:

```python
{
    "problem": "If a store has 45 apples and sells 17...",
    "answer": "51",
}
```

No solution examples needed - model generates its own!

### Configuration

```python
num_generations = 4      # Solutions per problem
temperature = 0.8        # Sampling for diversity
lora_r = 32             # Higher rank for reasoning
learning_rate = 5e-6     # Lower for stability
max_steps = 50
```

### Reward Function

Critical component - verifies solution correctness:

```python
def reward_function(completion, correct_answer):
    predicted = extract_answer(completion)
    if predicted == correct_answer:
        return 1.0  # Correct
    else:
        return 0.0  # Incorrect
```

Can be binary (0/1) or continuous (0-1) for partial credit.

### Training Process

For each problem:
1. Generate 4 different solution attempts
2. Extract answer from each
3. Compare with correct answer
4. Reward correct solutions (1.0)
5. Penalize incorrect solutions (0.0)
6. Update model to favor correct reasoning paths

Over time, success rate improves as model learns effective reasoning strategies.

### Output Files

```
./grpo_reasoning_smollm2/
├── adapter_config.json
├── adapter_model.bin
└── [Config files]

./smollm2_grpo_merged/
└── [Merged reasoning model]
```

### Usage

#### Testing Reasoning Ability

```python
def solve_problem(problem_text):
    prompt = f"Solve step by step:\nProblem: {problem_text}\n\nSolution:"
    
    outputs = model.generate(
        tokenizer(prompt, return_tensors="pt").to("cuda"),
        max_new_tokens=300,
        temperature=0.3,  # Lower for focused reasoning
    )
    
    return tokenizer.decode(outputs[0])

# Test
solution = solve_problem("A bakery made 96 cupcakes...")
print(solution)
```

### When to Use GRPO

Recommended For:
- Math problem solving (GSM8K, MATH dataset)
- Code generation with test cases
- Logic puzzles
- Multi-step reasoning tasks
- Tasks with verifiable answers
- Want chain-of-thought behavior

Not Recommended For:
- Open-ended generation
- Creative writing
- Tasks without clear correct answers
- Limited compute budget
- Real-time applications

### Reward Function Design

#### Binary Rewards

Simple, clear signal:
```python
def binary_reward(output, answer):
    return 1.0 if extract(output) == answer else 0.0
```

#### Continuous Rewards

Partial credit possible:
```python
def continuous_reward(output, answer):
    if fully_correct(output, answer):
        return 1.0
    elif partially_correct(output, answer):
        return 0.5
    else:
        return 0.0
```

#### Process Rewards

Reward intermediate steps:
```python
def process_reward(output, solution_steps):
    score = 0.0
    for step in solution_steps:
        if step in output:
            score += 0.2
    return min(score, 1.0)
```

### Troubleshooting

Issue: Success rate not improving
- Increase number of generations (N=8)
- Simplify problems to start
- Check reward function is correct
- Train for more steps
- Increase temperature for diversity

Issue: Model generates incoherent reasoning
- Lower temperature (0.6-0.7)
- Provide better prompting
- Ensure base model has reasoning capability
- May need supervised fine-tuning first

Issue: Expensive to train
- Reduce num_generations (N=2-4)
- Use smaller model
- Train on fewer problems initially
- Use mixed precision

### Best Practices

1. Start with easy problems, gradually increase difficulty
2. Ensure reward function is reliable
3. Generate diverse solutions (temperature > 0.7)
4. Monitor success rate during training
5. Test on held-out problems
6. Combine with supervised fine-tuning for best results

### Recommended Datasets

Math Reasoning:
- GSM8K: Grade school math word problems
- MATH: Competition mathematics
- AQuA: Algebraic word problems

Code:
- HumanEval: Python programming problems
- MBPP: Basic Python programming
- CodeContests: Competitive programming

Logic:
- LogiQA: Logical reasoning
- ARC: AI2 Reasoning Challenge
- bAbI: Reasoning tasks

### Extensions

#### Curriculum Learning

```python
# Start easy, gradually harder
easy_problems = problems[:100]
medium_problems = problems[100:300]
hard_problems = problems[300:]

train_grpo(model, easy_problems)
train_grpo(model, medium_problems)
train_grpo(model, hard_problems)
```

#### Ensemble Rewards

```python
def ensemble_reward(output, answer):
    # Multiple verification methods
    r1 = exact_match(output, answer)
    r2 = numerical_close(output, answer)
    r3 = logical_valid(output, answer)
    
    return max(r1, r2, r3)  # Any method works
```

### Resources

- GRPO Concept: OpenAI o1 System Card
- GSM8K Dataset: https://github.com/openai/grade-school-math


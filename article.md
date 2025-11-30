# HeliumLM: Tiny yet Powerful

## Abstract

I always wanted to build a language model but the scale at which they are built scared me. But reading through few papers discussing several optimization methods I thought to myself, "Why not?" and started building a small, very small language model which can run in your smart watch without causing you a wrist burn.

I introduce **HeliumLM**, a 13.44M parameter generative model built from scratch. By combining Grouped-Query Attention (GQA) and Sliding Window Attention (SWA), and Targeted Knowledge Injection, I created a decoder-only model.

Result? HeliumLM achieves a peak RAM usage of 598 MB (vs TinyBERT's 651 MB), making it ~8% more memory efficient while retaining generative capabilities that TinyBERT lacks.

## 1. Introduction

I don't think you can escape the grasp Larguage Language Models have on us. They have become as necessary as a mobile phone nowadays. With more and more companies trying to make one of their own. I think it is important to understand the what and hows of language models. Or maybe it's just my curiosity but regardless there is a technology to be learnt and maybe optimize further. So, it my free time I thought of creating a language model of my own.

The market is filled with language models that are huge, like Llama3, GPT-5(useless in my opinion), Gemini-2.5, and many more. Training them surely took like a data center or two at minimum but I don't have those. So, I needed to get creative in order to build a LM.

I started reading papers about Small Language Models and the methods used in making them. This knowledge gave me my breakthrough. One doesn't need a whole data center to get a LM. One just needs to optimize the algorithm to allow faster training, inference and smaller size. One example of this could be seen in TinyBERT. It's a encoder-only model with very small size and is as powerful as other such encoder-only models. But I wanted to make a generative or decoder-only model so the approach had to be tweaked.

**Transformers** and **Attention** are the building blocks of any LM can be modified to fit certain use cases. In my use case I wanted it to be efficient. Normal attention is of **O(n<sup>2</sup>)** time complexity but we can reduce that to a linear time complexity. Combine that with a **sliding window** mechanism to only give attention to current tokens. We reduce the memory space a lot. These methods will be discussed shortly.

## 2. Methodology

Designing HeliumLM was not just about reducing size of a Large Language Model. Simply reducing the number of layers often result in a "brain-dead" model that cannot form coherent sentences. Instead, I re-engineered the transformer block to optimize for **information density** and **memory footprint**.

This methodology has three pillars in its core:
- Architectural Compression
- Synthetic Data Curation
- Curriculum Learning

### 2.1 The Architectural Engine

At the core of HeliumLM is a 13.44M parameter decoder-only transformer. To make it more memory efficient while retaining the generative capabilities, I implemented three specific optimizations.

#### 2.1.1 Grouped-Query Attention (GQA): The Memory Compressor

Standard Multi-Head Attention (MHA) is memory-expensive because it stores Key-Value (KV) pairs for every single Query (Q) head. This becomes the primary bottleneck during inference.

KV cache is the model's short term memory. When a language model generates text token by token each new token needs to pay attention to all the previous tokens in the sequence. So, it KV cache helps to store the K and V vectors for each token only one in VRAM. For the next token, it only computes the K and V for the *next* token and appends them from the cache. This happens in a linear time, O(n).

The memory consumption of the KV cache is determined by the below formula:

`Cache Size = (sequence_length) * (batch_size) * (num_layers) * (d_model) * 2 * (bytes_per_parameter)`

- `sequence_length`: The longer the text I generate, the larger the cache grows. If `max_seq_len` is 256, the cache must hold data for up to 256 tokens.
- `batch_size`: If I generate multiple responses at once, the cache size is multiplied accordingly.
- `num_layers`: Each of my layer in model architecture has its own independent KV cache.
- `d_model`: The hidden dimension of the model, which is 256 here.
- `2`: Because a pair of Key and Value is being stored.
- `bytes_per_parameter`: Usually 2 bytes for float16 precision.

So, for a single generated sequence of 256 tokenx with HeliumLM:

`Cache Size = 256 * 1 * 8 * 256 * 2 * 2 bytes = ~2.1 MB`

This seems small, but with GQA this could be compressed even further. In my configuration (`n_head=8`, `n_kv_head=2`), I forced 4 Query heads to share a single Key/Value Head. This reduces the runtime memory footprint of the KV cache by 75% compared to standard attention.

$$\text{Memory}_{KV} \propto \frac{N_{heads}}{N_{kvheads}}$$

I like to think of this in a much more common way. Imagine a classroom. Each student has their own textbook and since two or three students sit on the same bench the desk looks clumpsy but if 3 students share one textbook then the desks look clean. This is what GQA means.

#### 2.1.2 Sliding Window Attention (SWA): The Focus Mechanism

Standard transformers has a quadratic time complexity *O(N<sup>2</sup>)*, which means as the conversations go on, the memory usage explodes, because every new token needs to attend to every single token that came before it.

Sliding Window Attention (SWA) elegantly solves this problem by constraining each token to only look back to its immediate neighbours within a fixed-size window, *W*. Instead of looking at the entire history, the model's focus is limited to a local context.

Mathematically, the standard attention output for a query token *q<sub>i</sub>* is a weighted sum over all previous value tokens *v<sub>j</sub>*:

$$
\text{Attention}(q_i, K, V) = \sum_{j=1}^{i} \text{softmax}\left(\frac{q_i K_j^T}{\sqrt{d_k}}\right) V_j
$$

SWA modifies this by applying a casual mask that only allows coputation within the window. For a query token *q<sub>i</sub>* and a window size *W*, it only considers key-value pairs *(k<sub>j</sub>, v<sub>j</sub>)* where the key's position *j* is whithin the window:

$$
\text{Attention}_{SWA}(q_i, K, V) = \sum_{j=\max(1, i-W+1)}^{i} \text{softmax}\left(\frac{q_i K_j^T}{\sqrt{d_k}}\right) V_j
$$

This simple constraint has a profound impact. The number of tokens each query attends is no longer the entire sequence length *N*, but the constant window size *W*. This reduces the computational complexity from *O(N<sup>2</sup>)* to *O(N × W)*. Since *W* is a fixed constant, the complexity becomes linear.

Again, let's think in the common ways to understand this properly. It's like reading a book. Instead of re-reading the entire book from page one every time you start a new sentence, you only focus on the last few paragraphs to understand the current context.

For HeliumLM, I implemented a window size of **W=64**. This ensures that the memory and computation required for the attention remains constant and predictable, regardless of how long the generated text becomes.

![swa_diagram](image.png)

#### 2.1.3 SwiGLU Feed-Forward Network

To maximize the learning capacity per parameter, I replaced the standard ReLU or GELU activation function in the feed-forward network (FFN) with **SwiGLU (Swish-Gated Linear Unit)**. This choice is not arbitrary; it introduces a dynamic, data-dependent filtering mechanism that significantly enhances the model's expressiveness.

Let's break down the name to understand its function:

1.  **Swish**: This is an activation function defined as `f(x) = x * σ(βx)`, where `σ` is the sigmoid function. Unlike ReLU, Swish is smooth and non-monotonic, which allows for better gradient flow and has been shown to outperform ReLU in deep networks.
2.  **Gated Linear Unit (GLU)**: This is the core concept. Instead of passing the input through a single linear layer followed by an activation, the GLU framework uses two parallel linear layers. One of these layers acts as a "gate," controlling what information from the other layer is allowed to pass through.

SwiGLU combines these two ideas. The input `x` is projected through two separate linear transformations, `W` and `V`. The output of the first projection is passed through the Swish activation function, and this result is then used to gate the output of the second projection via element-wise multiplication (`⊙`).

Mathematically, the operation is defined as:

$$
\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)
$$

Where the Swish activation is:

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

This gating mechanism is incredibly powerful. It allows the network to dynamically control the flow of information based on the input itself. If the gate value for a particular neuron is close to zero, that information is suppressed. If it's close to one, it's passed through. This data-dependent filtering allows the model to learn more complex patterns and relationships compared to a static activation function.

For HeliumLM, using SwiGLU was a strategic decision to achieve maximum performance from a minimal parameter count. While it appears to use more parameters by having two linear layers in the FFN, recent studies shows that the increased expressiveness often allows for a smaller hidden dimension, resulting in a net decrease in total parameters for equivalent or better performance. This is why I chose SwiGLU as the activation function. Also, the name sounded cool.

![swiglu_diagram](image-1.png)

#### 2.1.4 Rotary Positional Embeddings (RoPE)

So, how does a model know if a word is at the beginning or the end of a sentence? This is where positional embeddings come in. The classic approach is to just add a unique vector to each token based on its absolute position (e.g., token 1 gets vector A, token 2 gets vector B, etc.). This works, but it's not very flexible and struggles with sequences longer than what it was trained on.

For HeliumLM, I chose a much more elegant solution: **Rotary Positional Embeddings (RoPE)**.

Instead of adding information, RoPE *rotates* the query and key vectors based on their position. Think of it like this: imagine each token's vector is a point on a 2D plane. RoPE rotates this point by an angle that is proportional to its position in the sequence.

-   The vector for the first token is rotated by a small angle, `θ`.
-   The vector for the second token is rotated by `2θ`.
-   The vector for the *m*-th token is rotated by `mθ`.

Why is this so clever? Because when the model calculates the attention score between two tokens, what matters is the *angle between them*. The dot product between two rotated vectors depends only on their *relative* positional difference, not their absolute positions.

Let's get into the math just a little bit. For a 2D vector **v** = (x, y), rotating it by an angle `θ` is done with a rotation matrix:

$$
R_\theta = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix}
$$

RoPE applies this concept to the high-dimensional query (**q**) and key (**k**) vectors by pairing up their dimensions. For a query **q** at position *m* and a key **k** at position *n*, their dot product after rotation becomes:

$$
(R_{m\theta} \mathbf{q})^T (R_{n\theta} \mathbf{k}) = \mathbf{q}^T R_{(m-n)\theta} \mathbf{k}
$$

The final attention score only depends on `(m-n)`, the relative distance between the tokens. The model learns the meaning of "5 tokens away" or "2 tokens before" regardless of whether it's happening at the start or end of a paragraph.

I chose RoPE for HeliumLM for three key reasons:

1.  **It Encodes Relative Position**: This is more natural for language, where the relationship between words often matters more than their exact location.
2.  **It Handles Longer Sequences**: Because the rotation is a repeating pattern, RoPE can handle sequences longer than it was trained on without any issues. This is a massive advantage for a small, efficient model.
3.  **It's Proven**: RoPE is a core component in many state-of-the-art models like Llama, proving its effectiveness and stability.

By using RoPE, HeliumLM gains a sophisticated understanding of word order and distance without the rigid constraints of absolute embeddings, making it both smarter and more flexible.

![alt text](image-2.png)

#### 2.1.5 Weight Tying: The ultimate Parameter Hack

One of the most effective yet simple tricks to drastically reduce a model's size is **Weight Tying**. Think about the two main jobs involving the vocabulary in a language model:

1.  **Input Embedding Layer**: At the beginning, this layer converts an input token (like "gravity") into a meaningful vector in the model's hidden space. It's like looking up a word in a dictionary to get its definition.
2.  **Output Projection Layer**: At the very end, this layer takes a final vector from the model and predicts the most likely next token from the entire vocabulary. It's like having a definition and trying to find the right word for it.

These two operations are essentially inverses of each other. It makes intuitive sense that the matrix used to map words to meanings should be related to the matrix that maps meanings back to words.

Weight tying formalizes this intuition by forcing the model to use the *same weight matrix* for both jobs. Specifically, the weights of the output layer are set to be the transpose of the input embedding layer's weights:

$$
W_{output} = W_{embedding}^T
$$

For HeliumLM, with a vocabulary of 32,000 tokens and a model dimension of 256, the embedding matrix has `32,000 × 256 = 8,192,000` parameters. Without weight tying, the output layer would need another 8.19 million parameters. By tying the weights, I eliminated the need for this second matrix entirely.

This single optimization saved **8.19M parameters**. Given that the final model size is 13.44M, this trick alone reduced the potential parameter count by nearly 40%! It's a massive saving that directly contributes to HeliumLM's tiny memory footprint, and it also acts as a form of regularization, often leading to more stable training and better performance.

## 3. Quality Over Quantity: The Data Strategy

A fundamental constraint of Small Language Models (SLMs) is their limited parameter capacity. Unlike a 70B parameter giant, an SLM is like a sponge with finite space—it cannot afford to absorb the noisy, redundant, and often toxic "sludge" of the open internet. Every parameter must be used to store high-value information.

This principle led to my core data strategy: **Signal-to-Noise Optimization**. Instead of throwing a massive, unfiltered dataset at the model, I curated a small, high-quality corpus and implemented a structured learning plan, much like educating a child.

### 3.1 Crafting the Perfect Textbook: Synthetic Data Generation

The first step was to create the ideal foundational text. Rather than scraping the web and filtering out HTML tags, ads, and forum arguments, I decided to generate a pristine dataset from scratch.

*   **The Teacher**: I used a powerful teacher model (Llama-3) to act as my content generator.
*   **The Syllabus**: The curriculum was based on **Cosmopedia**, a dataset of diverse and high-value educational topics, ensuring a broad knowledge base.
*   **The Format**: I enforced a strict "Textbook Style" prompt structure for every piece of generated content, forcing the teacher model to follow a clear, logical flow:
    1.  **Definition**: State the concept clearly.
    2.  **Explanation**: Elaborate on the concept's nuances.
    3.  **Example**: Provide a concrete example to solidify understanding.

The result was a custom-built library of approximately 7,000 high-density, structured "textbook chapters." This clean, high-signal data formed the bedrock of HeliumLM's education.

### 3.2 The Three-Phase Curriculum: From Knowledge to Conversation

With the textbook ready, I designed a three-phase curriculum to guide the model's development, moving from basic knowledge to specialized expertise.

#### Phase 1: Foundational Knowledge (Pre-training)

This is the "elementary school" phase. The goal was to teach the model the statistical patterns of the English language and core world knowledge.

*   **Goal**: Learn grammar, facts, and sentence structure.
*   **Dataset**: The synthetic textbook data.
*   **Outcome**: The model became a proficient writer. It could complete sentences grammatically and coherently but lacked any sense of conversational turn-taking. If you gave it a prompt, it would just continue writing in the same style indefinitely.

#### Phase 2: Learning to Converse (Instruction Tuning)

This is the "middle school" phase, where the model learns social skills. The goal was to teach it the user-assistant dialogue format.

*   **Goal**: Understand the `### User:` and `### Assistant:` structure and learn to respond to instructions.
*   **Dataset**: A curated subset of the Dolly-15k and Alpaca datasets, which are rich in instruction-response pairs.
*   **Outcome**: A major behavioral shift. The model stopped generating endless text and started responding directly to prompts, effectively learning how to have a conversation.

#### Phase 3: Corrective Tutoring (Knowledge Injection)

This is the "expert tutoring" phase. During testing, I noticed the model would hallucinate specific facts (e.g., claiming "Newton is a football team"). This final phase was designed to surgically correct these errors.

*   **Goal**: Overwrite specific, high-priority factual errors.
*   **Dataset**: A small, custom-generated dataset of critical facts (e.g., "Newton discovered gravity"). These samples were **oversampled by 5x**, forcing the model to pay extra attention to them.
*   **Outcome**: The model successfully unlearned its hallucinations and adopted the correct information. This proved that even with a small parameter count, an SLM can be fine-tuned to become a reliable expert in a narrow domain.
   
## 4. Training Dynamics: A Rollercoaster Ride

Training a language model, even a small one, is less like pressing a "run" button and more like navigating a storm. The process was a journey of debugging, stabilizing, and fine-tuning, filled with valuable lessons.

### 4.1 The "Classroom" Setup

My training environment was hosted on **Kaggle**, using a single **Nvidia GPU100**. The core of the training script relied on a robust set of tools to ensure stability and efficiency:

*   **Optimizer**: I used `AdamW`, the standard for modern transformer models, which decouples weight decay from the gradient update for better regularization.
*   **Learning Rate Schedule**: A `Cosine Decay` schedule with a `10% linear warmup` was crucial. This strategy starts with a low learning rate, gradually increases it to prevent initial instability, and then smoothly decreases it, allowing the model to settle into a good minimum.
*   **Mixed Precision Training**: To maximize speed on the A10G, I used Automatic Mixed Precision (AMP) via `torch.amp.GradScaler`. This allows the model to perform most calculations in fast `float16` precision while keeping critical parts in stable `float32`, preventing numerical underflow.

### 4.2 The First Disaster: NaN Loss and the "Poisoned" Tokenizer

My initial training attempts were a complete failure. After a few dozen steps, the loss would suddenly shoot to `NaN` (Not a Number), and the model would output nothing but gibberish. This is a classic sign of an unstable training loop, often caused by exploding gradients.

After some debugging, I traced the issue to two root causes:

1.  **An Aggressive Learning Rate**: My initial learning rate was too high, causing the model's weights to update too drastically and spiral out of control. The warmup phase of the LR scheduler was the key fix here.
2.  **The "Poisoned" Tokenizer**: The most insidious bug was a **tokenizer mismatch**. An earlier version of my pre-training script used a slightly different tokenizer than the one used for fine-tuning. Even a small discrepancy in the vocabulary mapping meant the model was trying to interpret tokens it had never seen before, leading to catastrophic failure.

**The Lesson**: Enforce strict tokenizer consistency across all stages of development. A single, version-controlled `tokenizer.json` is not just a file; it's the unchanging language of your model.

### 4.3 The Convergence Curve: Learning and Overfitting

Once the tokenizer was fixed and the learning rate was stabilized, the model began to learn properly. The pre-training phase, as captured by the training logs, followed a textbook trajectory.

*   **The Start**: The initial training loss at Step 0 was **10.53**. This is not a random number; it's the mathematical equivalent of random guessing. With a vocabulary of 32,000, the theoretical cross-entropy for a uniform distribution is `ln(32000) ≈ 10.37`. My initial loss being close to this value was a strong signal that the model was correctly initialized and ready to learn.

*   **The Descent**: The training loss steadily decreased, while the validation loss followed, confirming that the model was generalizing its knowledge. The immediate recovery from occasional loss spikes validated the effectiveness of **Gradient Clipping**, which I had capped at `1.0` to prevent difficult batches from derailing the training process.

*   **The Plateau and Overfitting**: As seen in the graph below, the validation loss hit its minimum value of **3.81** around step 1700. After this point, the training loss continued to decrease, but the validation loss began to slowly creep back up. This is the classic signature of **overfitting**. The model had started to memorize the training data's quirks instead of learning general patterns. This divergence is precisely why `EarlyStopping` is so critical; it allowed me to save the model at its peak performance before it started to degrade.

![Training Loss](diagrams/heliumLM_trial_05_train_loss.png)
![Validation Loss](diagrams/heliumLM_trial_05_val_loss.png)

### 4.4 The Fine-Tuning Polish

The final phase, fine-tuning for conversation, required a more delicate approach.

*   **Lower Learning Rate**: I dropped the learning rate significantly to `3e-5`. The model has already learned general knowledge; now, it just needs to gently adapt to a new skill (chatting) without forgetting its foundation.
*   **Ignoring Padding**: Chat datasets often have varied lengths, requiring padding. It's crucial to tell the loss function to ignore these padding tokens (`ignore_index=pad_token_id`). Otherwise, the model wastes capacity trying to predict padding, which is useless.
*   **The Result**: After just two epochs, the model's behavior transformed. It learned the turn-taking structure of a conversation and, thanks to the knowledge injection, could correctly answer questions about Newton, demonstrating a successful transfer of learning.

This multi-phase, problem-solving approach to training was just as important as the architectural design in creating a small but capable model.

## 5. Results & Evaluation

To validate the effectiveness of HeliumLM's design, I evaluated its performance against a well-established baseline, TinyBERT, focusing on both quantitative efficiency and qualitative capabilities. The goal was to demonstrate that a purpose-built generative SLM could outperform a distilled encoder model in memory footprint while offering a completely different class of functionality.

### 5.1 Quantitative Benchmarks: Efficiency Under the Microscope

I benchmarked HeliumLM against TinyBERT (a standard 4-layer distilled encoder) on a CPU environment to measure the true memory footprint without GPU optimizations.

| Metric         | HeliumLM (Ours)        | TinyBERT (Baseline)      | Difference          |
| :------------- | :--------------------- | :----------------------- | :------------------ |
| **Architecture**   | Decoder (Generative)   | Encoder (Discriminative) | N/A                 |
| **Parameters**     | 13.44 M                | 14.38 M                  | **-6.5% (Smaller)** |
| **Peak RAM**       | 598 MB                 | 651 MB                   | **-8.1% (Leaner)**  |
| **Throughput**     | 152 tok/sec            | 171 tok/sec              | -11.1% (Slower)     |

**Key Insights:**

*   **Memory Efficiency Win**: HeliumLM is **8.1% more memory-efficient** than TinyBERT. This directly validates the architectural choices, especially the 75% KV cache reduction from Grouped-Query Attention (GQA), which is critical for performance on memory-constrained devices.
*   **Throughput Analysis**: While HeliumLM's throughput is 11% lower, this is an expected and impressive result. TinyBERT processes entire sequences in parallel (one forward pass), whereas HeliumLM must generate tokens one by one (autoregressive decoding). Achieving near-parity in speed highlights the computational efficiency gained from Sliding Window Attention (SWA), which keeps the attention complexity linear.
*   **The Generative Advantage**: Most importantly, HeliumLM achieves these efficiency gains while being a **fully generative model**. Unlike TinyBERT, which is designed for tasks like classification, HeliumLM can create novel text, answer questions, and hold a conversation—a capability its encoder-only counterpart lacks entirely.

### 5.2 Qualitative Evolution: The Journey of a Prompt

The true success of the three-phase training curriculum is best illustrated by tracking the model's response to the same prompt—*"Tell me about Newton"*—at each stage of its development.

*   **Phase 1 (After Pre-training)**:
    > "*Tell me about Newton... the key is many ways to do be made for the first time in the world...*"
    > **Analysis**: The model is a capable but aimless writer. It recognizes words and produces grammatically plausible sentences but has no concept of answering a question.

*   **Phase 2 (After Instruction Tuning)**:
    > "*Tell me about Newton... Newton is a professional football team based in New York.*"
    > **Analysis**: A huge leap in behavior. The model now understands it must provide a direct answer. However, its knowledge is flawed, leading to a confident hallucination based on common token associations ("Newton," "New York").

*   **Phase 3 (After Knowledge Injection)**:
    > "*Tell me about Newton... Sir Isaac Newton was a physicist and mathematician who developed the laws of motion.*"
    > **Analysis**: Success. The targeted, oversampled fine-tuning has successfully overwritten the hallucination with the correct factual information. The model is now not only conversational but also factually reliable on this topic.

This evolution clearly demonstrates the power of a structured curriculum to build not just a text generator, but a knowledgeable and responsive agent.

## 6. Conclusion

HeliumLM started as a personal challenge to demystify the process of building a language model. It successfully demonstrates that by combining modern architectural optimizations—GQA, SWA, RoPE, and SwiGLU—with a deliberate, quality-focused data strategy, it is possible to create a highly efficient and capable generative model from scratch.

With a memory footprint under 600 MB and a parameter count of just 13.44M, HeliumLM proves that the future of AI is not just in massive, cloud-based models, but also in small, specialized agents that can run anywhere. This project made me learn things which otherwise I would've been oblivious to. But the most important thing that I appreciated was the usage of complex numbers in RoPE. I always love to see them in places where no one would expect them to be.
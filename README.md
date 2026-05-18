# Mechanistic Interpretability via Typoglycemic Perturbations

Typoglycemia is the phenomenon where words with scrambled interior letters remain readable to
humans — *"the huamn mnid deos not raed ervey lteter"*. This project uses that property as a
controlled experimental probe on GPT-style transformer models.

Matched pairs of normal and typoglycemic text are fed to the model, and deviations in cosine
similarity and WordNet relatedness between paired outputs become entry points for mechanistic
investigation. The question is not whether the model performs well or poorly, but what the
pattern of performance reveals about the model's internals.

This project is part of a larger sequential modelling project:
[sequential modelling ](https://github.com/Dorcas-Joy-Kahunguka/sequence-modelling)

---

## The probe

### Semantic accuracy via cosine similarity and WordNet relatedness

Matched plaintext and typoglycemic inputs are passed through the model. Outputs are compared
using cosine similarity and WordNet relatedness to assess whether the model generates semantically
equivalent responses.

**Assumption:** the model produces meaningfully similar outputs for typoglycemic text and its
plaintext equivalent.

**If the assumption holds — similar generations:**

The model is resolving the surface distortion somewhere in its forward pass. The mechanistic
investigation asks where convergence occurs:

- Are the two inputs tokenised differently? If tokenisation already produces similar token
  sequences, resolution may be happening before the transformer layers entirely.
- If tokenised differently, do the token embeddings converge? Comparing embedding vectors for
  typoglycemic and plaintext tokens establishes whether the embedding layer has learned
  form-invariant representations.
- If embeddings diverge, the investigation moves into the transformer layers — probing attention
  head activations and MLP outputs layer by layer to locate where the representations converge.
  Which heads are active at the point of convergence? What do they appear to have learned?

**If the assumption does not hold — dissimilar generations:**

The model treats the two inputs as carrying different meaning. The mechanistic question shifts
from locating convergence to locating divergence:

- Do the token embeddings start close and diverge at some layer, or do they start far apart and
  remain so throughout?
- If divergence is introduced at a specific layer, which components are responsible — attention
  heads, MLP layers, or both?
- What does this tell us about the level of abstraction at which the model commits to an
  interpretation of input? This outcome may reflect a meaningful difference between human
  top-down contextual processing and transformer sequence processing.

---

## Pipeline

```
1. Dataset construction
   Collect plaintext sentence pairs.
   Generate typoglycemic equivalents via interior-letter scrambling.

2. Behavioural evaluation
   Run both versions through the model.
   Compute cosine similarity between output token embeddings.
   Compute WordNet relatedness between the words decoded from the top predicted tokens.
   Visualise the similarity distribution across the paired dataset.

3. Hypothesis resolution
   Assess whether the assumption holds, partially holds, or fails.
   Each outcome produces a directed mechanistic question.

4. Mechanistic investigation
   Probe tokenisation, token embeddings, attention heads, MLP layers, and residual stream
   activations at the point of interest identified in step 3.
   Use activation patching to isolate which components drive convergence or divergence.

5. Findings
   Characterise what the model has or has not learned about surface-form-invariant meaning.
```

---

## Structure

```
03-typoglycemia-interpretability/
├── data/
│   ├── raw/                        # Source word lists or sentence corpora
│   └── processed/
│       ├── plaintext.jsonl         # Plaintext inputs
│       └── typoglycemic.jsonl      # Scrambled equivalents, aligned by index
├── notebooks/
│   ├── 01-dataset-construction.ipynb
│   ├── 02-behavioural-evaluation.ipynb
│   ├── 03-tokenisation-analysis.ipynb
│   ├── 04-embedding-comparison.ipynb
│   ├── 05-attention-probing.ipynb
│   └── 06-findings.ipynb
├── src/
│   ├── scramble.py                 # Interior-letter scrambling with controls
│   ├── data.py                     # Dataset loading and pair alignment
│   ├── evaluate.py                 # Cosine similarity and WordNet relatedness
│   ├── probe.py                    # Activation extraction and patching via TransformerLens
│   └── visualise.py                # Similarity distributions, attention head heatmaps
├── configs/
│   └── experiment.yaml             # Model name, layer range, probe settings
├── outputs/
│   ├── figures/                    # Similarity plots, attention head visualisations
│   └── logs/                       # Per-run evaluation results
├── tests/
│   ├── test_scramble.py
│   ├── test_evaluate.py
│   └── test_probe.py
├── checkpoints/                    # Cached model weights (git-ignored)
│   └── .gitkeep
├── .gitignore
├── README.md
├── environment.yml
└── requirements.txt
```

---

## Notes

- Target model: GPT-2 or comparable open-weights GPT-style transformer.
- Interpretability tools: TransformerLens for activation patching and head probing.
- This project does not train a model from scratch. It investigates the internals of a
  pretrained model using the transformer architecture knowledge built in
  [seq-models-02-transformers](https://github.com/Dorcas-Joy-Kahunguka/seq-models-02-transformers).
- Both outcomes of each assumption are treated as findings. This is not a hypothesis that
  can only be confirmed — it is a controlled perturbation designed to produce informative
  contrast regardless of direction.


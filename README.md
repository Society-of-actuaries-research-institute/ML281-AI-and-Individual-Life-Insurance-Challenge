# ML281-AI-and-Individual-Life-Insurance-Challenge-John-Koestner
AI and Life Insurance Challenge - Submission from John Koestner

## 1 Overview

The SOA had a competition in 2026 where the participants were to demonstrate how AI can complement traditional actuarial techniques, enhance understanding of experience data, and contribute to better decision-making across pricing, valuation, and risk management.

The dataset was the [ILEC 12-19 dataset](https://cdn-files.soa.org/research/ilec/ilec-mort-text-data-dict-2012-2019.zip)

**Research Objectives**
Submissions should balance attention to methodological approach and practical insights. Specifically, participants are expected to:

- Justify the choice of AI techniques, including considerations of model design, transparency, interpretability, and appropriateness for actuarial practice.
- Demonstrate what practitioners can learn from the application of these methods, highlighting strengths and limitations compared to traditional actuarial techniques.
- Communicate insights drawn from the dataset, including trends, anomalies, or actionable findings that could inform real-world actuarial decision-making.
- Present results in a way that is understandable to both technical and non-technical stakeholders.

## 2 Jupyter Lab

### 2.1 Jupyter Lab Usage

The following notebook provides an example workflow for modeling mortality using neural network.

**🔬 Jupyter Notebook:**

- [Neural Network Model with Sampled Data](http://githubtocolab.com/Society-of-actuaries-research-institute/ML281-AI-and-Individual-Life-Insurance-Challenge-John-Koestner/blob/main/notebooks/soa_neural.ipynb)

A full data notebook has been added to this repository to show similar results to the sampled data notebook, however more resources than a standard laptop are needed to analyze. A GPU is recommended.


### 2.2 Local Install
To install, download this repository and run the following command in 
the environment of choice.

```
conda create --name soa python=3.12
pip install -r requirements.txt
```

To have conda environments work with Jupyter Notebooks a kernel needs to be 
defined. This can be done defining a kernel, shown below.

`python -m ipykernel install --user --name=soa`

## 3 AI assisted analysis (CLAUDE)

Claude (an AI assistant) has the ability to use a repository and generate analysis. A "skill" file has been created so that Claude can use the workflow that was run in the `soa_neural.ipynb` notebook. The skill files are located in `.claude\skills\nn-review`.

### 3.1 Claude Usage

Once in the terminal Claude will automatically invoke this skill when you ask things like:
  - "Review my neural network model using the nn-review skill and output results here"
  - "Analyze the A/E ratios for my mortality model using the nn-review skill and output results here"
  - "Generate SHAP analysis for my predictions using the nn-review skill and output results here"
  - "Check if my model is overfitting using the nn-review skill and output results here"
  
It's not necessary to add "using the nn-review skill", however it does ensure the model behaves as expected.

The skill includes preferred patterns like using amount_exposed as weights, death_count as secondary axis, and the morai libraries.

### 3.2 Install
To use claude the CLI terminal should be installed as documented below. This does require an API to use.

https://code.claude.com/docs/en/quickstart

## 4 Hardware Recommendations

For the sampled data the following specs are recommended
- CPU is fine - 12 cores
- RAM: 16GB
- Storage: 30GB

For the full data the following specs are recommended
- GPU is recommended - (NVIDIA RTX A4500 or better)
- VRAM: 20GB
- Storage: 30GB



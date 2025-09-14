# 📘 DataPrivacyLLM - PETS

A framework for experimenting with data privacy in Large Language Models (LLMs).

---

## 🚀 Getting Started

Run the main entry point:

```bash
python main.py
```

### Experiments are defined in:

```
src/configs/config.py
```

Copy the desired configuration into `main.py` based on the experiment you want to run.

### Create a new virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

### Analysis of results can be found in:

```
combined_analysis_deception.ipynb
combined_analysis_missing_information.ipynb
```

---

# Artifact Appendix (for PETS Artifact Evaluation)

Paper title: **Sanitization or Deception? Rethinking Privacy Protection in Large Language Models**

Requested Badge(s):
- [x] **Available**
- [ ] Functional
- [ ] Reproduced

---

## Description

This artifact accompanies the paper *"Sanitization or Deception? Rethinking Privacy Protection in
Large Language Models"*.  
It contains the source code and analysis scripts used in our experiments on **data sanitization and privacy in Large Language Models (LLMs)**.  

The repository provides:
1. Core framework (`main.py`, `src/`) for running sanitization experiments.  
2. Configuration files (`src/configs/config.py`) for specifying experiment parameters.  
3. Analysis notebooks (`combined_analysis_missing_information.ipynb`) for reproducing the evaluation plots and tables in the paper.  

---

## Security/Privacy Issues and Ethical Concerns

- The code itself does **not** disable security features or run vulnerable code.  
- No exploits, malware, or sensitive datasets are included.  
- Experiments are based on public datasets or synthetic data formatted as expected by the framework.  
- There are **no human subjects or user study data** bundled in this artifact.  

---

## Environment

### Accessibility
The artifact is publicly available at:  
**https://github.com/BipinPaudel/PrivacyOrDeception_PETS**

### Hardware Requirements
- Can run on a laptop (no special hardware requirements) given that it can run the required LLM models.  
- GPU is optional but can accelerate LLM evaluations.  

### Software Requirements
- Tested on **Ubuntu 22.04** and **macOS 15.3.1**.  
- Python **3.10+** required.  
- All other dependencies are listed in `requirements.txt`.  

To set up:

```bash
git clone <your-repo-link>
cd <project-folder>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Testing the Environment

To verify that the setup works, run:

```bash
python main.py
```

This should execute a default experiment with a simple configuration and produce output logs under `results/`.  
If successful, you can modify configurations in `src/configs/config.py` to run different experiments.

---

## Limitations

This artifact is released under the **Available** badge.  
It provides all code and analysis scripts, but does **not** guarantee reproducibility of full experimental results without further computational resources (e.g., access to large LLMs).  

---

## Notes on Reusability

The framework can be reused to test other text-sanitization methods or extended to new datasets and adversaries.  
Users can modify:  
- `src/configs/config.py` for new experiments.  
- `analysis notebooks` for evaluating additional privacy–utility trade-offs.  

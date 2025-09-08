# Code Repository for *Advancing Quantum Imaging through Learning Theory*

This repository contains the source code used to generate all results presented in the paper *Advancing Quantum Imaging through Learning Theory*. The scripts reproduce every figure in both the main text and the Supplemental Material.

The scripts are organized to reproduce every figure in both the main text and the Supplemental Material:

- **Main Text Figures**: `main_fig*.py`  
- **Supplemental Material Figures**: `SM_fig*.py`  

Expected output: plots corresponding to the figures in the paper.
Expected runtime: less than 12 hours on a normal desktop computer.
To reproduce figures in the main text, run the corresponding main_figX.py script.
To reproduce figures in the Supplemental Material, run the corresponding SM_figX.py script.
Each script is self-contained and can be executed directly to reproduce the corresponding figure.
No external dataset is required; each script generates all necessary data internally.

## System requirements
- Programming language: Python 3.12  
- Dependencies: NumPy (v1.26.4), SciPy (v1.13.1), Matplotlib (v3.9.2), sklearn (v1.5.1)
- Non-standard hardware: None (runs on a normal desktop/laptop computer)  
- Operating systems tested: Windows 11
  
## Installation guide
Clone this repository and install the required dependencies:  
```bash
git clone https://github.com/ykwang-phys/quantum-learning-imaging
cd quantum-learning-imaging

Typical installation time: less than 5 minutes on a normal desktop computer.









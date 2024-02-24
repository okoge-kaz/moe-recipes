# MoE Recipes

# Table of Contents

1. [Installation](#installation)

## Installation

To install the package, run the following command:

```bash
pip install -r requirements.txt
```

If you want to use the library in multi-nodes, you need to install the below packages:

```bash
module load openmpi/4.x.x

pip install mpi4py
```

### FlashAttention

To install the FlashAttention, run the following command: (GPU is required)

```bash
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

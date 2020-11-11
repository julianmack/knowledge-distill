# Knowledge Distillation

For current results and training commands see [Results Notebook](notebooks/Results.ipynb).

## Setup and Installation

1. Install deps with miniconda:

   ```bash
   conda env create -f environment.yml
   conda activate distill
   pip install -e .
   ```

2. Download GloVe embeddings. The Glove directory can be in any location but the
default is `./model`.

3. Download (or train) a teacher `fast-berst` model and place contents in `./model`.

4. Place csv data for training/evaluation in `./data` directory.

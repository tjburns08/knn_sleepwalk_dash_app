# KNN Sleepwalk Dash App

## What this is

The general problem is that nonlinear dimensionality reduction tools can be misleading. Here, the user is able to get better intuition around the "resolution" of their embeddings by viewing the location of the K-nearest neighbors (KNN) in "embedding space" versus the "original feature space." The latter could be surface markers if you're dealing with flow/CyTOF data, or PCAs if you are dealing with single-cell sequencing data.

## Credit, and origin of the project

This is a UI around my KnnSleepwalk project. This was originally an R package, which you can find [here](https://github.com/tjburns08/KnnSleepwalk). The project was inspired by the sleepwalk app from Anders Biostat which can be found [here](https://anders-biostat.github.io/sleepwalk/). While that app focused on looking at distances, I focus on nearest neighbors. The original package was a fork of their package, which manipulated the distance matrix and visualization color scheme to get out the KNN. 

Here, I re-wrote it from scratch so it would be more UI friendly and more easily modifiable on my end, as I build this out.

## Use the web app

For lightweight usage (if datasets are larger than 50k cells, you'll have to subsample down to 50k), you can use the web app. This can be found [here](https://knn-sleepwalk-dash-app.plotly.app/)

## Installation

Go into the command line, and follow the instructions below.

```bash
# clone
git clone https://github.com/tjburns08/knn_sleepwalk_dash_app.git
cd knn_sleepwalk_dash_app

# create and activate a virtual env
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux

# if Windows (PowerShell): 
.\.venv\Scripts\Activate.ps1

# install required packages
pip install -r requirements.txt

# run
python app.py
```
Then, open the printed URL (e.g., http://127.0.0.1:8050)

## How to use

When you click on the printed URL, a UI interface will pop up. At the top are the instructions. But I'll repeat them here:

1. Upload **Original Markers** CSV (whatever you used to generate the embedding) and **Embedding (UMAP/t-SNE, etc)** CSV (we assume same row count & order). Use "max cells" to automatically downsample large uploads (default 20k). This helps us get around an issue with the web version, where it gets buggy when you upload very large datasets.

2. Click **Run Knn Sleepwalk**. If no files are uploaded, the example dataset loads.

3. Use the **k** slider (default 25) to change neighbor count.

4. Hover a point to highlight its kNN: **left** = UMAP-space kNN, **right** = original-features kNN.

## Additional questions and feedback

The maintainer of the project is Tyler Burns. His LinkedIn is [here](https://www.linkedin.com/in/tylerjburns/): 

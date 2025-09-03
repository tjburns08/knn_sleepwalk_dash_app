# linked_hover_tsne_knn_fast.py
from dash import Dash, dcc, html, Input, Output, State, Patch, no_update, ctx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import base64, io

MAX_K = 50

# ---------------- helpers ----------------

def load_example():
    # works whether files have an index col or not — we’ll align by order later
    orig = pd.read_csv("../data/example_surface_markers.csv")
    dimr = pd.read_csv("../data/example_umap.csv")
    return orig, dimr

def parse_upload(contents: str, filename: str | None):
    """Decode dcc.Upload CSV -> pandas.DataFrame (no index required)."""
    if not contents:
        return None
    try:
        _, b64 = contents.split(",", 1)
        buf = io.StringIO(base64.b64decode(b64).decode("utf-8"))
        df = pd.read_csv(buf)
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse '{filename or 'uploaded file'}': {e}")

def _drop_pseudo_index(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a first column that looks like an exported index (e.g., 'Unnamed: 0' or 0..n-1)."""
    if df.shape[1] == 0:
        return df
    first = df.columns[0]
    # drop 'Unnamed: 0' style
    if isinstance(first, str) and first.lower().startswith("unnamed"):
        return df.drop(columns=[first])
    # drop 0..n-1 integer index column
    s0 = df.iloc[:, 0]
    if pd.api.types.is_integer_dtype(s0) and s0.reset_index(drop=True).equals(pd.Series(range(len(df)))):
        return df.drop(columns=[first])
    return df

def prepare_aligned_ordered(orig_df: pd.DataFrame, dimr_df: pd.DataFrame):
    """
    Align by row order (no IDs). Keep numeric columns only.
    Require: same number of rows in both CSVs.
    Drop any row that has NaN in ORIG (any feature) or the first two dims of DIMR.
    Return (orig, dimr, x, y).
    """
    o = _drop_pseudo_index(orig_df).select_dtypes(include=[np.number])
    d = _drop_pseudo_index(dimr_df).select_dtypes(include=[np.number])

    if len(o) != len(d):
        raise ValueError(f"Row counts differ: features={len(o)} rows, embedding={len(d)} rows. "
                         f"Please upload CSVs with the same number of rows (same order).")

    if d.shape[1] < 2:
        raise ValueError("Embedding CSV needs at least two numeric columns (x,y).")

    # mask rows that are fully usable in BOTH tables (order-aligned)
    good_o = ~o.isna().any(axis=1)
    good_d = ~d.iloc[:, :2].isna().any(axis=1)
    good = (good_o & good_d).to_numpy()

    if good.sum() < len(o):
        # silently drop bad rows from BOTH to preserve alignment
        o = o.loc[good].copy()
        d = d.loc[good].copy()

    x = d.iloc[:, 0].to_numpy()
    y = d.iloc[:, 1].to_numpy()
    return o, d, x, y

def compute_knn(orig: pd.DataFrame, dimr_xy: np.ndarray):
    n = orig.shape[0]
    if n < 2:
        raise ValueError("Dataset must have at least 2 usable rows after cleaning.")
    kmax = min(MAX_K, n - 1)

    # LEFT: kNN in UMAP/tSNE space (first two dims)
    nbrs_left = NearestNeighbors(n_neighbors=kmax + 1, metric="euclidean").fit(dimr_xy[:, :2])
    knn_left_idx = nbrs_left.kneighbors(return_distance=False)[:, 1:]

    # RIGHT: kNN in original feature space
    nbrs_right = NearestNeighbors(n_neighbors=kmax + 1, metric="euclidean").fit(orig.values)
    knn_right_idx = nbrs_right.kneighbors(return_distance=False)[:, 1:]

    return knn_left_idx, knn_right_idx, kmax

# ---------- plotting helpers you need ----------
def base_fig(x, y, title):
    """Traces: [0]=cloud, [1]=neighbors overlay, [2]=center overlay."""
    cloud = go.Scattergl(
        x=x, y=y, mode="markers",
        marker=dict(size=4, opacity=0.35),
        #hovertemplate="cell %{pointIndex}<extra></extra>",
        hovertemplate="<extra></extra>",
        showlegend=False,
    )
    neighbors = go.Scattergl(
        x=[], y=[], mode="markers",
        marker=dict(size=8, opacity=0.95),
        hoverinfo="skip", showlegend=False,
    )
    center = go.Scattergl(
        x=[], y=[], mode="markers",
        marker=dict(size=12, opacity=1),
        hoverinfo="skip", showlegend=False,
    )
    fig = go.Figure(data=[cloud, neighbors, center])
    fig.update_layout(
        title=title, height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="keep", dragmode="pan", hovermode="closest",
    )
    return fig

def center_and_neighbors(idx, knn_mat, k):
    """Return [center_idx], [center + its k neighbors] for overlaying."""
    if idx is None:
        return [], []
    nbh = knn_mat[idx][:k]
    return [idx], [idx, *list(nbh)]

def clear_patches():
    """Empty the neighbor/center overlays on both figures."""
    p1, p2 = Patch(), Patch()
    for p in (p1, p2):
        p["data"][1]["x"], p["data"][1]["y"] = [], []
        p["data"][2]["x"], p["data"][2]["y"] = [], []
    return p1, p2


# ---------------- init empty ----------------
xv0, yv0 = np.array([]), np.array([])
knn_left0 = np.empty((0, 0), dtype=int)
knn_right0 = np.empty((0, 0), dtype=int)

# ---------------- app ----------------
app = Dash(__name__)
app.title = "KNN Sleepwalk!"
app.layout = html.Div(
    [
        html.H1(
            "Welcome to KNN Sleepwalk!",
            style={"margin":"0 0 12px 0", "techAlign":"center"}
        ),

        html.Div(
            dcc.Markdown(
                """\
**What this is:**

Embeddings can sometimes be misleading. Here, walk through your embedding to see exactly how the nearest neighbors in embedding space compare to the nearest neighbors in feature space. I recommend you do this with every new dataset and embedding.

**How to use:**

1. Upload **Original Markers** CSV (whatever you used to generate the embedding) and **Embedding (UMAP/t-SNE, etc)** CSV (we assume same row count & order). If the dataset is greater than 5000 cells, it will subsample down to 5000. We intend to add larger dataset functionality in later versions.

2. Click **Run Knn Sleepwalk**. If no files are uploaded, the example dataset loads.

3. Use the **k** slider (default 25) to change neighbor count.

4. Hover a point to highlight its kNN: **left** = UMAP-space kNN, **right** = original-features kNN.

**Want more?**

Visit my website [here](https://tjburns08.github.io/) where I write, research, and build in the open. Also visit my [company website](https://burnslsc.com/), my [LinkedIn](https://www.linkedin.com/in/tylerjburns/) and my [GitHub](https://github.com/tjburns08/),
"""
            ),
            style={
                "background": "#f8f9fa",
                "border": "1px solid #e5e7eb",
                "borderRadius": "8px",
                "padding": "10px 12px",
                "margin": "0 0 12px 0",
                "fontSize": "20px",
            },
        ),

        dcc.Store(id="last", data={"idx": None, "k": None}),
        dcc.Store(id="xy", data={"x": xv0.tolist(), "y": yv0.tolist()}),
        dcc.Store(id="knn-left", data=knn_left0.tolist()),
        dcc.Store(id="knn-right", data=knn_right0.tolist()),

        html.Div(
            [
                html.Div(
                    [
                        html.Div("1) Upload surface markers csv file."),
                        dcc.Upload(
                            id="upload-orig",
                            children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
                            multiple=False, accept=".csv,text/csv",
                            style={"width":"100%","height":"60px","lineHeight":"60px",
                                   "borderWidth":"1px","borderStyle":"dashed",
                                   "borderRadius":"8px","textAlign":"center"}
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("2) Upload embedding (e.g. t-SNE, UMAP) csv file."),
                        dcc.Upload(
                            id="upload-dimr",
                            children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
                            multiple=False, accept=".csv,text/csv",
                            style={"width":"100%","height":"60px","lineHeight":"60px",
                                   "borderWidth":"1px","borderStyle":"dashed",
                                   "borderRadius":"8px","textAlign":"center"}
                        ),
                    ]
                ),
            ],
            style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}
        ),

        html.Div(
            [
                html.Button("Run Knn Sleepwalk", id="run-btn", n_clicks=0,
                            style={"padding":"8px 14px","borderRadius":"8px",
                                   "border":"1px solid #999","cursor":"pointer"}),
                html.Span(id="upload-status", style={"marginLeft":"10px","fontSize":"12px","color":"#666"}),
            ],
            style={"margin":"8px 0 10px 0"}
        ),

        html.Label("k nearest neighbors"),
        dcc.Slider(id="k", min=1, max=50, step=1, value=25, tooltip={"placement":"bottom"}),

        html.Div(
            [
                dcc.Graph(
                    id="plot1",
                    figure=base_fig([], [], "UMAP layout — kNN in UMAP space"),
                    clear_on_unhover=False,
                ),
                dcc.Graph(
                    id="plot2",
                    figure=base_fig([], [], "UMAP layout — kNN in feature space"),
                    clear_on_unhover=False,
                ),
            ],
            style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"},
        ),
    ],
    style={"maxWidth":"1200px","margin":"0 auto","padding":"8px"}
)

# -------- run button: choose uploads if both present else example --------
@app.callback(
    Output("knn-left", "data"),
    Output("knn-right", "data"),
    Output("xy", "data"),
    Output("plot1", "figure", allow_duplicate=True),
    Output("plot2", "figure", allow_duplicate=True),
    Output("last", "data",   allow_duplicate=True),
    Output("upload-status", "children"),
    Output("k", "max"),
    Output("k", "value"),
    Input("run-btn", "n_clicks"),
    State("upload-orig", "contents"),
    State("upload-dimr", "contents"),
    State("upload-orig", "filename"),
    State("upload-dimr", "filename"),
    State("k", "value"),
    prevent_initial_call=True,
)
def on_run(n_clicks, c_orig, c_dimr, f_orig, f_dimr, k_current):
    try:
        use_uploads = bool(c_orig and c_dimr)
        if use_uploads:
            df_orig = parse_upload(c_orig, f_orig)
            df_dimr = parse_upload(c_dimr, f_dimr)

            # HACK adding subsampling step
            n = len(df_orig)
            sub_size = 5000
            seed = 42
            if n > sub_size:
                rng = np.random.default_rng(seed)
                keep = np.sort(rng.choice(n, sub_size, replace = False))
                df_orig = df_orig.iloc[keep].reset_index(drop = True)
                df_dimr = df_dimr.iloc[keep].reset_index(drop = True)

            orig, dimr, x, y = prepare_aligned_ordered(df_orig, df_dimr)
            source_msg = f"using uploads ({f_orig} + {f_dimr})"
        else:
            orig_ex, dimr_ex = load_example()
            orig, dimr, x, y = prepare_aligned_ordered(orig_ex, dimr_ex)
            source_msg = "using built-in example"

        dimr_xy = dimr.iloc[:, :2].values
        knn_left, knn_right, kmax = compute_knn(orig, dimr_xy)

        # replace clouds & clear overlays; autorange
        p1, p2 = Patch(), Patch()
        for p in (p1, p2):
            p["data"][0]["x"], p["data"][0]["y"] = x.tolist(), y.tolist()
            p["data"][1]["x"], p["data"][1]["y"] = [], []
            p["data"][2]["x"], p["data"][2]["y"] = [], []
            p["layout"]["xaxis"]["autorange"] = True
            p["layout"]["yaxis"]["autorange"] = True

        k_new = min(int(k_current or 8), kmax)
        status = f"Loaded {orig.shape[0]} rows; k ∈ [1,{kmax}] — {source_msg}"

        return (
            knn_left.tolist(),
            knn_right.tolist(),
            {"x": x.tolist(), "y": y.tolist()},
            p1, p2,
            {"idx": None, "k": None},
            status,
            kmax,
            k_new,
        )
    except Exception as e:
        return (no_update, no_update, no_update, no_update, no_update,
                no_update, f"Run error: {e}", no_update, no_update)

# -------- hover: draw neighbors using current stores --------
@app.callback(
    Output("plot1", "figure", allow_duplicate=True),
    Output("plot2", "figure", allow_duplicate=True),
    Output("last", "data",   allow_duplicate=True),
    Input("plot1", "hoverData"),
    Input("plot2", "hoverData"),
    Input("k", "value"),
    State("knn-left", "data"),
    State("knn-right", "data"),
    State("last", "data"),
    State("xy", "data"),
    prevent_initial_call=True,
)
def update_on_hover(hover1, hover2, k, knn_left, knn_right, last, xy):
    # no data yet
    if not xy or not xy.get("x"):
        p1, p2 = clear_patches()
        return p1, p2, {"idx": None, "k": None}

    trig = ctx.triggered_id
    if trig == "plot1":
        idx = hover1["points"][0]["pointIndex"] if hover1 else None
    elif trig == "plot2":
        idx = hover2["points"][0]["pointIndex"] if hover2 else None
    else:
        idx = last.get("idx")

    xv = np.asarray(xy["x"])
    yv = np.asarray(xy["y"])

    if idx is None or idx >= len(xv):
        p1, p2 = clear_patches()
        return p1, p2, {"idx": None, "k": None}

    k = int(k)
    # guard in case slider > available after a rerun
    k_avail = min(len(knn_left[idx]), len(knn_right[idx]))
    k = min(k, k_avail)

    if last and last.get("idx") == idx and last.get("k") == k:
        return no_update, no_update, last

    left_center, left_all   = center_and_neighbors(idx, knn_left,  k)
    right_center, right_all = center_and_neighbors(idx, knn_right, k)

    p1, p2 = Patch(), Patch()
    p1["data"][1]["x"], p1["data"][1]["y"] = xv[left_all].tolist(),  yv[left_all].tolist()
    p1["data"][2]["x"], p1["data"][2]["y"] = xv[left_center].tolist(), yv[left_center].tolist()

    p2["data"][1]["x"], p2["data"][1]["y"] = xv[right_all].tolist(),  yv[right_all].tolist()
    p2["data"][2]["x"], p2["data"][2]["y"] = xv[right_center].tolist(), yv[right_center].tolist()

    return p1, p2, {"idx": idx, "k": k}

# ---------------- main ----------------
if __name__ == "__main__":
    app.run(debug=True)

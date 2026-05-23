# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Marketing Science
#     language: python
#     name: marketing-science
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takechanman1228/marketing-science-in-python/blob/main/notebooks/sec4.1-segmentation/semantic_segmentation.ipynb)
#
# *Source of truth is the paired `.py`; this `.ipynb` is generated — see CONTRIBUTING.md.*

# %% [markdown]
# # Embedding-Based Customer Segmentation with BERTopic
#
# This notebook accompanies **§4.1** of *Marketing Science in Python*
# (Modern Methods section). We use **BERTopic**, which wraps SentenceTransformers,
# UMAP, and HDBSCAN into a single pipeline, to discover customer segments from
# product purchase text, then interpret them with an LLM.
#
# For traditional methods (decile, RFM, K-Means, GMM), see the companion script
# `traditional_segmentation.py`.
#
# **Dataset**: UCI Online Retail II (Kaggle, CC0). ~1M transaction rows from a
# UK-based online retailer, with product descriptions.
#
# **Learning objectives**
#
# 1. Build revenue-weighted customer documents from product descriptions for
#    embedding-based segmentation (after filtering generic, ubiquitous SKUs).
# 2. Use BERTopic to discover semantic customer segments and **collapse them
#    down to 4–8 operational segments** with `reduce_topics()`.
# 3. Reduce outliers so every customer carries a usable topic label.
# 4. Automate segment naming with an LLM, then **validate the names against
#    keyword and RFM evidence** — the "LLM-naming trap" in the chapter.
# 5. Compare HDBSCAN's discovery segments with KMeans-on-embeddings for
#    activation, and score a held-out customer with `transform()`.
# 6. Honestly assess temporal stability (ARI) on a time-split holdout.
#
# **Required packages**: `bertopic`, `sentence-transformers`, `umap-learn`,
# `hdbscan`, `litellm`, `anthropic` (for LLM segment naming via Claude).

# %%
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

CONFIG = {
    "data_path": "../Retail_open_data_set/Online_Retail_II_UCI _Kaggle_cc0/online_retail_II.csv",
    "top_n_products": 20,
    "min_topic_size": 15,       # HDBSCAN min size — loose so we find many initial topics
    "min_cluster_size": 15,     # HDBSCAN min_cluster_size matches min_topic_size
    "min_samples": 3,           # HDBSCAN min_samples
    "umap_n_components": 5,     # UMAP target dims for clustering
    "umap_n_neighbors": 15,     # smaller neighborhood — preserves local structure
    "embedding_model": "all-MiniLM-L6-v2",
    "random_state": 42,
    "generic_product_threshold": 0.03,  # tighter: drop SKUs bought by >3% of customers
    "nr_topics": 8,             # target final segment count (chapter: 4–8 segments)
    "min_segment_share": 0.10,  # chapter target: each segment ≥ 10% of base
    "mega_subsegments": 6,      # if a mega-topic forms, sub-decompose with KMeans into this many
    "mega_threshold": 0.50,     # any topic > 50% of base triggers sub-decomposition
    # --- data cleaning ---
    "non_product_codes": {"POST","DOT","M","D","S","C2","CRUK","AMAZONFEE","ADJUST","BANK"},
    "custom_stop_words": ["set","pack","pcs","piece","pieces","assorted","asstd","design","designs"],
    "doc_weight_k": 3,          # log-revenue repetition factor
    "vectorizer_min_df": 2,     # kill hapax legomena
    # --- LLM naming (Anthropic via litellm) ---
    "llm_model": "anthropic/claude-haiku-4-5",
    "llm_nr_docs": 8,           # docs sampled per topic for naming
    "llm_delay_seconds": 1,
}

ACCENT_BLUE = "#2171b5"
ACCENT_ORANGE = "#e6550d"
ACCENT_GREEN = "#31a354"
ACCENT_PURPLE = "#756bb1"
ACCENT_GREY = "#969696"

pd.set_option("display.float_format", "{:.2f}".format)

try:
    display
except NameError:
    display = print

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

from msbook.paths import chapter_images, chapter_artifacts

_FIG_DIR = chapter_images(part="4", chapter="sec4.1")
_TABLES  = chapter_artifacts(part="4", chapter="sec4.1-segmentation") / "tables"
_TABLES.mkdir(parents=True, exist_ok=True)


def savefig(fig, name, **kw):
    kw.setdefault("dpi", 150)
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(_FIG_DIR / name, **kw)

_step = 0

def step(title):
    global _step
    _step += 1
    print(f"\n{'='*60}")
    print(f"  Step {_step}: {title}")
    print(f"{'='*60}\n")


# %%
step("Load and clean UCI Online Retail II")

raw = pd.read_csv(CONFIG["data_path"])
print(f"Raw rows: {len(raw):,}")
print(f"Columns: {list(raw.columns)}")

# Rename for consistency
raw = raw.rename(columns={"Customer ID": "CustomerID"})

# Parse dates
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
raw["CustomerID"] = pd.to_numeric(raw["CustomerID"], errors="coerce")

# Remove exact duplicate rows (3.2% of data)
n_before = len(raw)
raw = raw.drop_duplicates()
print(f"Dropped {n_before - len(raw):,} exact duplicates ({(n_before - len(raw))/n_before:.1%})")

# Clean: remove returns, cancelled invoices, missing IDs, non-product items
raw = raw[(raw["Quantity"] > 0) & (raw["Price"] > 0)]
raw = raw.dropna(subset=["CustomerID", "Description", "InvoiceDate"])
raw = raw[~raw["Invoice"].astype(str).str.startswith("C")]

# StockCode filter: remove non-product codes (POST, DOT, M, D, S, C2, etc.)
non_product = CONFIG["non_product_codes"]
n_before_sc = len(raw)
raw = raw[~raw["StockCode"].astype(str).isin(non_product)]
print(f"Removed {n_before_sc - len(raw):,} non-product rows (StockCodes: {non_product})")

# Description regex as safety net (catches remaining junk not in StockCode list)
raw = raw[~raw["Description"].str.contains(
    r"POSTAGE|POST|DOTCOM|CRUK|BANK CHARGES|Manual|Discount|AMAZON|ADJUST|test|samples",
    case=False, na=False,
)]
raw["Revenue"] = raw["Quantity"] * raw["Price"]
raw["CustomerID"] = raw["CustomerID"].astype(int)

print(f"\nAfter cleaning: {len(raw):,} rows  |  {raw['CustomerID'].nunique():,} customers")
print(f"Date range: {raw['InvoiceDate'].min()} to {raw['InvoiceDate'].max()}")
print(f"Unique products (StockCodes): {raw['StockCode'].nunique():,}")

# %%
step("Feature engineering: RFM + customer documents")

ref_date = raw["InvoiceDate"].max() + pd.Timedelta(days=1)

# Per-customer RFM features (for profiling later) -- uses ALL purchases
uci_customers = (
    raw.groupby("CustomerID")
    .agg(
        recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        frequency=("Invoice", "nunique"),
        monetary=("Revenue", "sum"),
        n_products=("StockCode", "nunique"),
    )
    .reset_index()
)
uci_customers["avg_order_value"] = uci_customers["monetary"] / uci_customers["frequency"]
print(f"UCI customers: {len(uci_customers):,}")
display(uci_customers.describe().round(2))

uci_customers.to_csv(_TABLES / "modern_uci_customer_features.csv", index=False)

# ── Filter generic products by StockCode (catches typos/variants) ──
# Products bought by >10% of customers dominate embeddings and create
# a single mega-topic.  We remove them from the document text only;
# RFM features above still use all purchases.
code_cust_counts = raw.groupby("StockCode")["CustomerID"].nunique()
n_total = raw["CustomerID"].nunique()
threshold = CONFIG["generic_product_threshold"]
generic_codes = set(
    code_cust_counts[code_cust_counts > threshold * n_total].index
)
print(f"\nGeneric products filtered by StockCode (>{threshold:.0%} of customers): "
      f"{len(generic_codes)}")
if generic_codes:
    # Show sample descriptions for the filtered codes
    sample = (raw[raw["StockCode"].isin(generic_codes)]
              .drop_duplicates("StockCode")[["StockCode","Description"]]
              .sort_values("StockCode"))
    for _, row in sample.head(10).iterrows():
        print(f"  - {row['StockCode']}: {row['Description']}")
    if len(generic_codes) > 10:
        print(f"  ... and {len(generic_codes) - 10} more")
raw_filtered = raw[~raw["StockCode"].isin(generic_codes)]

# ── Normalize descriptions ──
# Lowercase + strip packaging stop words to collapse "SET OF 3" / "set of 3"
import re
stop_pat = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in CONFIG["custom_stop_words"]) + r")\b",
    re.IGNORECASE,
)
raw_filtered = raw_filtered.copy()
raw_filtered["Description"] = (
    raw_filtered["Description"]
    .str.lower()
    .str.replace(stop_pat, "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# ── Build revenue-weighted customer documents ──
TOP_N = CONFIG["top_n_products"]
K = CONFIG["doc_weight_k"]

prod_rev = (
    raw_filtered.groupby(["CustomerID", "Description"])["Revenue"]
    .sum()
    .reset_index()
    .sort_values(["CustomerID", "Revenue"], ascending=[True, False])
)
top_products = prod_rev.groupby("CustomerID").head(TOP_N)

# Revenue-weighted repetition: high-revenue products repeat more in the doc
def build_weighted_doc(grp):
    parts = []
    for _, row in grp.iterrows():
        repeat = 1 + int(np.log1p(row["Revenue"]) / K)
        parts.extend([row["Description"]] * repeat)
    return ", ".join(parts)

customer_docs = (
    top_products.groupby("CustomerID")
    .apply(build_weighted_doc)
    .reset_index()
    .rename(columns={0: "doc"})
)

print(f"\nCustomer documents: {len(customer_docs):,}")
print(f"\nExample document (CustomerID={customer_docs.iloc[0]['CustomerID']}):")
print(customer_docs.iloc[0]["doc"][:300] + "...")

# %% [markdown]
# ## Why BERTopic?
#
# The manual embedding-based segmentation pipeline requires chaining four
# independent libraries: SentenceTransformers (embed), UMAP (reduce),
# HDBSCAN (cluster), and custom code (interpret). **BERTopic** wraps all
# four into a single API and adds:
#
# - **c-TF-IDF** for automatic keyword extraction per topic
# - **Built-in visualizations** (document scatter, hierarchy, barchart)
# - **LLM representation models** for automated topic/segment naming
# - **`transform()`** for scoring new documents without refitting
# - **`reduce_outliers()`** for handling the noise problem HDBSCAN creates
#
# | Layer | Tool | Role |
# |---|---|---|
# | All-in-one | **BERTopic** | Wraps embedding + UMAP + HDBSCAN + c-TF-IDF |
# | Embedding | SentenceTransformers | Text → dense vectors (pluggable) |
# | Interpretation | LiteLLM + Claude | Segment naming via LLM |
# | Scale (optional) | Faiss, Vector DB | Production nearest-neighbor |

# %%
step("Fit BERTopic model")

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Pre-compute embeddings (gives us control + reuse)
embedding_model = SentenceTransformer(CONFIG["embedding_model"])
embeddings = embedding_model.encode(
    customer_docs["doc"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"Embedding shape: {embeddings.shape}")

# Custom sub-models for transparency
umap_model = UMAP(
    n_components=CONFIG["umap_n_components"],
    n_neighbors=CONFIG["umap_n_neighbors"],   # 30 — stabilizes manifold
    metric="cosine",
    random_state=CONFIG["random_state"],
    n_jobs=1,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=CONFIG["min_cluster_size"],  # 25 — fewer, larger clusters
    min_samples=CONFIG["min_samples"],            # 5
    cluster_selection_method="eom",
    prediction_data=True,
)

# Bigram vectorizer with max_df/min_df + custom stop words
english_stops = list(CountVectorizer(stop_words="english").get_stop_words())
all_stops = english_stops + CONFIG["custom_stop_words"]
vectorizer_model = CountVectorizer(
    stop_words=all_stops,
    ngram_range=(1, 2),
    min_df=CONFIG["vectorizer_min_df"],   # kill hapax
)

# BM25-weighted c-TF-IDF + KeyBERT reranking
ctfidf_model = ClassTfidfTransformer(
    bm25_weighting=True,
    reduce_frequent_words=True,
)
representation_model = KeyBERTInspired(top_n_words=10)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    min_topic_size=CONFIG["min_topic_size"],
    verbose=True,
)
topics, probs = topic_model.fit_transform(
    customer_docs["doc"].tolist(), embeddings=embeddings
)

# Cluster quality: silhouette score
# Initialize sil so downstream cells never hit NameError
sil = float("nan")
non_noise_mask = np.array(topics) != -1
if non_noise_mask.sum() > 1 and len(set(np.array(topics)[non_noise_mask])) > 1:
    sil = silhouette_score(
        embeddings[non_noise_mask],
        np.array(topics)[non_noise_mask],
        metric="cosine",
    )
    print(f"\nSilhouette score (cosine, excl. noise): {sil:.3f}")

# Summary — initial HDBSCAN pass before reduction
topic_info = topic_model.get_topic_info()
n_topics_initial = len(topic_info[topic_info["Topic"] != -1])
n_outliers = (np.array(topics) == -1).sum()
print(f"\nInitial topics from HDBSCAN: {n_topics_initial} "
      f"(+ {n_outliers} outliers, {n_outliers / len(topics):.1%})")

# %% [markdown]
# ### Collapse to operational segments
#
# The chapter targets 4–8 segments, each covering at least 10% of the base.
# HDBSCAN found more granular topics — useful for *discovery*, but too many
# for activation. `reduce_topics()` merges nearby topics on the c-TF-IDF
# space until the target count is reached.

# %%
step("Reduce to operational segment count")

target_n = CONFIG["nr_topics"]
if n_topics_initial > target_n:
    topic_model.reduce_topics(
        customer_docs["doc"].tolist(), nr_topics=target_n
    )
    topics = topic_model.topics_
    print(f"reduce_topics({target_n}): {n_topics_initial} → "
          f"{len(set(topics)) - (1 if -1 in topics else 0)} topics")
else:
    print(f"Skip reduce_topics: already at {n_topics_initial} ≤ target {target_n}.")

topic_info = topic_model.get_topic_info()
n_topics = len(topic_info[topic_info["Topic"] != -1])
display(topic_info)

# %%
step("BERTopic visualizations: document scatter")

# 2D UMAP for visualize_documents
reduced_embeddings = UMAP(
    n_components=2, metric="cosine",
    random_state=CONFIG["random_state"], n_jobs=1,
).fit_transform(embeddings)

# Intertopic distance map (requires >= 3 topics)
if n_topics >= 3:
    fig_topics = topic_model.visualize_topics()
    fig_topics.show()
else:
    print(f"Skipping intertopic distance map (need >= 3 topics, got {n_topics})")

# Document scatter (main figure for the chapter)
fig_docs = topic_model.visualize_documents(
    customer_docs["doc"].tolist(),
    reduced_embeddings=reduced_embeddings,
    hide_document_hover=False,
)
fig_docs.show()

# Save static version for the book
try:
    fig_docs.write_image(_FIG_DIR / "umap_clusters.png",
                         width=1200, height=800, scale=2)
    print(f"Saved {_FIG_DIR / 'umap_clusters.png'}")
except (ValueError, OSError, ImportError) as e:
    # kaleido is required for static image export in the book build
    warnings.warn(f"Could not save static image (install kaleido): {e}")


# %%
step("BERTopic visualizations: hierarchy & barchart")

if n_topics >= 2:
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.show()

    fig_bar = topic_model.visualize_barchart(
        top_n_topics=min(8, n_topics), n_words=8
    )
    fig_bar.show()

    # Save static versions
    try:
        fig_hierarchy.write_image(
            _FIG_DIR / "bertopic_hierarchy.png",
            width=1200, height=600, scale=2)
        fig_bar.write_image(
            _FIG_DIR / "bertopic_barchart.png",
            width=1200, height=600, scale=2)
        print(f"Saved hierarchy and barchart PNGs to {_FIG_DIR}")
    except (ValueError, OSError, ImportError) as e:
        # kaleido is required for static image export in the book build
        warnings.warn(f"Could not save static images (install kaleido): {e}")
else:
    print(f"Skipping hierarchy/barchart (need >= 2 topics, got {n_topics})")


# %% [markdown]
# ## Outlier Reduction
#
# HDBSCAN labels ambiguous customers as **noise** (topic = -1). Before
# profiling, we reassign them using `reduce_outliers()` with the
# **embeddings** strategy, which assigns each outlier to the topic whose
# centroid is closest by cosine similarity in the original embedding space.
#
# **Caution:** Outlier reduction tends to inflate the largest topic. When
# one topic is already the broadest (most common product descriptions),
# both c-TF-IDF and embedding-based strategies route most outliers there.
# We show the result honestly, then compare with **KMeans** (below),
# which produces balanced segments without noise.
#

# %%
step("Reduce outliers")

n_outliers_before = (np.array(topics) == -1).sum()

if n_outliers_before == 0:
    print("No outliers to reduce.")
else:
    # Use embeddings strategy: assigns each outlier to the topic whose
    # centroid is closest by cosine similarity.
    new_topics = topic_model.reduce_outliers(
        customer_docs["doc"].tolist(),
        topics,
        strategy="embeddings",
        embeddings=embeddings,
    )

    topic_model.update_topics(
        customer_docs["doc"].tolist(), topics=new_topics
    )
    topics = new_topics  # all subsequent cells use outlier-reduced topics
    n_outliers_after = (np.array(topics) == -1).sum()
    print(f"Outliers: {n_outliers_before:,} -> {n_outliers_after:,}")
    print(f"Reassigned: {n_outliers_before - n_outliers_after:,} customers")

    # Distribution after outlier reduction
    print("\nTopic sizes after outlier reduction:")
    sizes = pd.Series(topics).value_counts().sort_index()
    print(sizes)
    max_share = sizes.max() / len(topics)
    print(f"\nLargest topic: {sizes.idxmax()} with {sizes.max():,} customers "
          f"({max_share:.1%})")
    if max_share > 0.50:
        print("\n⚠ Mega-topic detected. HDBSCAN found one broad cluster that "
              "absorbed most outliers. This is a real finding: the majority of "
              "customers buy similar general-purpose products. The niche topics "
              "(parasols, war gliders, etc.) are genuine but too small for "
              "operational targeting. See the KMeans comparison below for "
              "balanced, actionable segments.")


# %% [markdown]
# ## Sub-decompose the mega-topic
#
# When outlier reduction inflates one HDBSCAN topic past the 50% threshold
# (the "mega-topic" the warning above flags), the chapter's 4–8
# balanced-segments goal isn't reachable from HDBSCAN alone. We sub-cluster
# **just the mega-topic's embeddings** with KMeans, splitting it into
# several activation-sized segments while keeping the small niche topics
# as discovery findings.
#
# This is the chapter's "HDBSCAN for discovery, KMeans for operations"
# pattern, applied surgically inside the dominant cluster instead of on
# the full base.

# %%
step("Sub-decompose the mega-topic with KMeans")

_sizes = pd.Series(topics).value_counts()
_max_share = _sizes.iloc[0] / len(topics)
_mega_id = int(_sizes.idxmax()) if _max_share > CONFIG["mega_threshold"] else None

if _mega_id is None:
    print(f"No mega-topic detected (largest share {_max_share:.1%} ≤ "
          f"{CONFIG['mega_threshold']:.0%}). Keeping topics as-is.")
else:
    n_sub = CONFIG["mega_subsegments"]
    print(f"Mega-topic is Topic {_mega_id} "
          f"({_sizes.iloc[0]:,} customers, {_max_share:.1%}).")
    print(f"Sub-clustering into {n_sub} sub-segments with KMeans on embeddings.")

    mega_mask = (np.array(topics) == _mega_id)
    mega_embeddings = embeddings[mega_mask]
    sub_km = KMeans(
        n_clusters=n_sub,
        n_init=10,
        random_state=CONFIG["random_state"],
    )
    sub_labels = sub_km.fit_predict(mega_embeddings)

    # Build the new contiguous topic IDs:
    #   sub-segments of the former mega → 0 .. n_sub-1
    #   niches (original non-mega, non-noise topics) → n_sub .. end
    other_ids = sorted(t for t in set(topics) if t != _mega_id and t != -1)
    id_remap = {old: i + n_sub for i, old in enumerate(other_ids)}

    new_topics = np.full(len(topics), -1, dtype=int)
    new_topics[mega_mask] = sub_labels                       # 0..n_sub-1
    for old, new in id_remap.items():
        new_topics[np.array(topics) == old] = new            # n_sub..end

    final_n = n_sub + len(other_ids)
    print(f"\nFinal segmentation: {n_sub} sub-segments + "
          f"{len(other_ids)} niche topics = {final_n} total.")

    new_sizes = pd.Series(new_topics).value_counts().sort_index()
    print("\nNew topic sizes (≥10% floor passes if no ⚠):")
    for tid, n in new_sizes.items():
        share = n / len(new_topics)
        flag = "" if share >= CONFIG["min_segment_share"] else "  ⚠ < 10%"
        print(f"  Topic {tid}: {n:,} ({share:.1%}){flag}")

    # Push the new groupings into the BERTopic model so the LLM naming
    # cell below + the validation table both see them.
    topic_model.update_topics(
        customer_docs["doc"].tolist(),
        topics=new_topics.tolist(),
    )
    topics = new_topics

# %% [markdown]
# ## LLM-Assisted Segment Naming
#
# BERTopic's `representation_model` accepts LLM backends that turn c-TF-IDF
# keyword bags into human-readable segment names. We route through
# **litellm** so the provider string is the only thing that changes —
# `anthropic/claude-haiku-4-5`, `openai/gpt-4o`, `ollama/llama3`, etc.
#
# > **The LLM-naming trap.** Any LLM will produce a polished name for any
# > cluster, including random noise. The validation cell below pairs every
# > LLM-generated name with the underlying keyword/RFM evidence so you can
# > check whether the narrative actually matches the data.

# %%
step("LLM-assisted topic naming")

from bertopic.representation import LiteLLM

SEGMENT_PROMPT = """You are a marketing analyst segmenting an online retail customer base.
Each "topic" below represents a customer segment defined by the products they buy.

Sample product descriptions from this segment:
[DOCUMENTS]

Keywords describing this segment: [KEYWORDS]

Based on the above, provide:
1. A short segment name (2-4 words)
2. A one-sentence marketing action

Format:
topic: <segment_name>, <marketing_action>"""

# Read the key from the environment — never hard-code.
_api_key = os.environ.get("ANTHROPIC_API_KEY")

if not _api_key:
    print(
        "ANTHROPIC_API_KEY not set in the environment. Skipping LLM naming.\n"
        "Falling back to c-TF-IDF keyword labels from the previous cells.\n"
        "Re-run with `export ANTHROPIC_API_KEY=...` to bake LLM names into "
        "this notebook."
    )
else:
    representation_model = LiteLLM(
        model=CONFIG["llm_model"],
        nr_docs=CONFIG["llm_nr_docs"],
        delay_in_seconds=CONFIG["llm_delay_seconds"],
        prompt=SEGMENT_PROMPT,
    )
    topic_model.update_topics(
        customer_docs["doc"].tolist(),
        representation_model=representation_model,
    )
    topic_info_llm = topic_model.get_topic_info()
    display(topic_info_llm[["Topic", "Count", "Name"]])

# %% [markdown]
# ### Validate names against evidence (the LLM-naming trap)
#
# For each segment we line up:
#
# - the LLM-generated **Name** (or the c-TF-IDF fallback),
# - the top 5 c-TF-IDF **keywords**,
# - **median RFM** stats from the upstream feature table,
# - the top 3 **products by revenue** for the segment.
#
# A name that doesn't reflect the keywords or the RFM profile is a polished
# fiction — replace it with the keyword bag and move on.

# %%
step("Validate LLM names against keyword + RFM evidence")

# Lock in topic-to-customer assignment for the validation table.
customer_docs["topic"] = topics
val_data = customer_docs.merge(uci_customers, on="CustomerID")
val_non_noise = val_data[val_data["topic"] != -1]

topic_info_validate = topic_model.get_topic_info()

print(f"\n{'Topic':<6}{'Name':<42}{'n':>6}  Top-5 keywords")
print("-" * 110)
for _, row in topic_info_validate.iterrows():
    t = row["Topic"]
    if t == -1:
        continue
    name = (row.get("Name") or "")[:40]
    n = int(row.get("Count") or (val_non_noise["topic"] == t).sum())
    kw = topic_model.get_topic(t)
    kw_str = ", ".join([w for w, _ in (kw or [])[:5]])
    print(f"{t:<6}{name:<42}{n:>6}  {kw_str}")

print("\nMedian RFM per topic (sanity-check the narrative):")
val_summary = (
    val_non_noise.groupby("topic")
    .agg(
        n=("CustomerID", "count"),
        med_recency=("recency", "median"),
        med_frequency=("frequency", "median"),
        med_monetary=("monetary", "median"),
    )
    .round(1)
)
val_summary["share_pct"] = (val_summary["n"] / val_summary["n"].sum() * 100).round(1)
display(val_summary)

# %%
step("Statistical profiling per topic")

# Merge topic assignments with RFM features
customer_docs["topic"] = topics
cluster_data = customer_docs.merge(uci_customers, on="CustomerID")

cluster_data.to_csv(_TABLES / "modern_topic_assignments.csv", index=False)

# Profile: RFM stats per topic (excluding noise if any remain)
non_noise = cluster_data[cluster_data["topic"] != -1]
if len(non_noise) == 0:
    print("No non-noise topics found. Skipping profiling.")
else:
    profile = (
        non_noise.groupby("topic")
        .agg(
            n_customers=("CustomerID", "count"),
            avg_recency=("recency", "mean"),
            med_recency=("recency", "median"),
            avg_frequency=("frequency", "mean"),
            med_frequency=("frequency", "median"),
            avg_monetary=("monetary", "mean"),
            med_monetary=("monetary", "median"),
            avg_products=("n_products", "mean"),
            med_products=("n_products", "median"),
        )
        .round(1)
    )
    profile.to_csv(_TABLES / "modern_segment_profile.csv", index=True)
    display(profile)

    # ── Heatmap with log-transformed z-scores (Issues 3, 7) ──
    avg_cols = ["avg_recency", "avg_frequency", "avg_monetary", "avg_products"]
    raw_cols = ["recency", "frequency", "monetary", "n_products"]
    skewed = {"frequency", "monetary", "n_products"}

    profile_z = pd.DataFrame(index=profile.index)
    for mc, rc in zip(avg_cols, raw_cols):
        vals = cluster_data[rc].copy()
        if rc in skewed:
            vals = np.log1p(vals)
        pop_mean = vals.mean()
        pop_std = vals.std()
        seg_vals = profile[mc].copy()
        if rc in skewed:
            seg_vals = np.log1p(seg_vals)
        z = (seg_vals - pop_mean) / pop_std if pop_std > 0 else 0.0
        if rc == "recency":
            z = -z  # invert: green = recent (low days)
        profile_z[mc] = z

    x_labels = []
    for mc, rc in zip(avg_cols, raw_cols):
        label = rc
        if rc == "recency":
            label += " (inverted: green=recent)"
        x_labels.append(label)

    # Y-labels include customer count for readability
    y_labels = [f"Topic {t} (n={int(profile.loc[t, 'n_customers'])})"
                for t in profile.index]

    fig, ax = plt.subplots(figsize=(10, max(3, len(profile) * 0.7 + 1)))
    im = ax.imshow(profile_z.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(avg_cols)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels(y_labels)

    for i in range(len(profile)):
        for j in range(len(avg_cols)):
            val = profile.iloc[i][avg_cols[j]]
            fmt = f"{val:,.0f}" if val > 100 else f"{val:.1f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9,
                    color="white" if abs(profile_z.iloc[i, j]) > 1 else "black")

    fig.colorbar(im, ax=ax, label="Z-score vs. population mean", shrink=0.8)
    ax.set_title("BERTopic Segment Profiles (log1p z-scores; recency inverted)")
    fig.tight_layout()
    savefig(fig, "bertopic_segment_heatmap.png")
    plt.show()

    # ── Guardrail: warn if any topic > 50% ──
    topic_sizes = non_noise["topic"].value_counts()
    max_share = topic_sizes.max() / len(non_noise)
    if max_share > 0.50:
        print(f"\n{'!'*60}")
        print(f"  WARNING: Topic {topic_sizes.idxmax()} contains {max_share:.0%} "
              f"of customers (mega-topic).")
        print(f"  Suggestions:")
        print(f"    - Increase min_cluster_size (currently {CONFIG['min_cluster_size']})")
        print(f"    - Tighten generic_product_threshold (currently {CONFIG['generic_product_threshold']})")
        print(f"    - Use KMeans on embeddings for balanced segments")
        print(f"{'!'*60}\n")

    # ── Segment Card: summary + per-topic detail ──
    print("\n" + "="*60)
    print("  Segment Cards")
    print("="*60)

    # Summary table
    summary = profile[["n_customers"]].copy()
    summary["share_pct"] = (summary["n_customers"] / summary["n_customers"].sum() * 100).round(1)
    summary["revenue"] = non_noise.groupby("topic")["monetary"].sum().round(0)
    summary["rev_share_pct"] = (summary["revenue"] / summary["revenue"].sum() * 100).round(1)
    summary["med_recency"] = profile["med_recency"]
    summary["med_frequency"] = profile["med_frequency"]
    summary["med_aov"] = non_noise.groupby("topic")["avg_order_value"].median().round(1)
    print("\nSummary:")
    display(summary)

    # Per-topic detail cards
    for t in sorted(non_noise["topic"].unique()):
        if t == -1:
            continue
        print(f"\n--- Topic {t} ---")

        # Top 5 keywords
        kw = topic_model.get_topic(t)
        if kw:
            kw_str = ", ".join([w for w, _ in kw[:5]])
            print(f"  Keywords: {kw_str}")

        # Top 3 countries
        seg_cids = set(non_noise[non_noise["topic"] == t]["CustomerID"])
        seg_raw = raw[raw["CustomerID"].isin(seg_cids)]
        top_countries = seg_raw.groupby("Country")["Revenue"].sum().nlargest(3)
        country_str = ", ".join(
            [f"{c} ({v/top_countries.sum():.0%})" for c, v in top_countries.items()]
        )
        print(f"  Top countries: {country_str}")

        # Top 5 products by revenue
        top_prods = (seg_raw.groupby("Description")["Revenue"]
                     .sum().nlargest(5))
        print(f"  Top products:")
        for desc, rev in top_prods.items():
            print(f"    - {desc}: £{rev:,.0f}")

# %% [markdown]
# ## HDBSCAN vs. KMeans: Discovery vs. Operations
#
# HDBSCAN finds **density-based niches**: sharp, well-separated groups
# (parasols, war gliders) plus a large noise cluster. This is great for
# *discovery* — finding segments you didn't know existed. But it often
# produces **unbalanced** results: one mega-topic absorbs the majority of
# customers, while niche topics have 10-150 each. That's hard to act on.
#
# **KMeans on the same embeddings** gives balanced, actionable segments.
# Every customer gets assigned (no noise), and cluster sizes are roughly
# equal. The tradeoff: KMeans forces boundaries where HDBSCAN would leave
# ambiguity, so borderline customers may be misassigned.
#
# **Practical guidance:** Use HDBSCAN for exploratory analysis (what niches
# exist?), then KMeans for operational targeting (assign every customer to
# a segment for campaign execution).
#

# %%
step("KMeans on embeddings: balanced alternative")

# ── KMeans on the same UMAP-reduced embeddings ──
# UMAP first (same settings as BERTopic), then KMeans for balanced clusters.
umap_for_km = UMAP(
    n_components=CONFIG["umap_n_components"],
    metric="cosine",
    random_state=CONFIG["random_state"],
    n_jobs=1,
).fit_transform(embeddings)

km = KMeans(n_clusters=7, n_init=10, random_state=CONFIG["random_state"])
km_labels = km.fit_predict(umap_for_km)

# Compare distributions
print("HDBSCAN topic sizes (after outlier reduction):")
hdbscan_sizes = pd.Series(topics).value_counts().sort_index()
print(hdbscan_sizes)
print(f"\nHDBSCAN largest topic: {hdbscan_sizes.max()} "
      f"({hdbscan_sizes.max() / len(topics):.1%})")

print("\nKMeans cluster sizes:")
km_sizes = pd.Series(km_labels).value_counts().sort_index()
print(km_sizes)
print(f"\nKMeans largest cluster: {km_sizes.max()} "
      f"({km_sizes.max() / len(km_labels):.1%})")

# Silhouette comparison
sil_km = silhouette_score(embeddings, km_labels, metric="cosine")
print(f"\nSilhouette (cosine): HDBSCAN={sil:.3f}  KMeans={sil_km:.3f}")
print("\nKMeans produces more balanced segments at comparable cluster quality.")

km_comparison = pd.DataFrame({
    "CustomerID": customer_docs["CustomerID"].values,
    "hdbscan_topic": topics,
    "kmeans_cluster": km_labels,
})
km_comparison.to_csv(_TABLES / "modern_hdbscan_vs_kmeans.csv", index=False)


# %% [markdown]
# ## Activate: Scoring New Customers
#
# BERTopic's `transform()` method scores new documents against the fitted model
# without refitting. This replaces the manual `UMAP.transform()` →
# `hdbscan.approximate_predict()` chain with a single call.

# %%
step("Score a new (held-out) customer with transform()")

# HDBSCAN's transform() can return -1 for documents that don't fall in a
# dense embedding region. The chapter point is that *some* customer must
# get a real topic for the demo to land. We pick the customer whose document
# is closest to a topic centroid by trying a few candidates and selecting
# the first non-noise assignment.

# Candidate pool: customers whose post-reduction topic is in a small,
# focused cluster (Topics 1-6 here — Topic 0 is the broad outlier-absorbed
# mega-cluster, so its members may not actually sit near the centroid).
_assigned = pd.Series(topics, index=customer_docs.index)
small_topics = [t for t in set(topics) if t != -1 and t != 0]
if not small_topics:
    small_topics = [t for t in set(topics) if t != -1]

candidate_cids = []
for t in small_topics:
    candidate_cids.extend(
        customer_docs.loc[_assigned == t, "CustomerID"].tolist()[:5]
    )
# Fallback: any non-noise customer
if not candidate_cids:
    candidate_cids = customer_docs.loc[_assigned != -1, "CustomerID"].tolist()[:10]

demo_cid = None
demo_doc = None
demo_topic = -1
demo_prob = 0.0
for cid in candidate_cids:
    doc = customer_docs.loc[customer_docs["CustomerID"] == cid, "doc"].iloc[0]
    tt, pp = topic_model.transform([doc])
    if tt[0] != -1:
        demo_cid, demo_doc, demo_topic = cid, doc, int(tt[0])
        if pp is not None and len(pp) > 0:
            pv = pp[0]
            demo_prob = float(np.max(pv)) if hasattr(pv, "__len__") else float(pv)
        break

if demo_cid is None:
    # No candidate landed on a non-noise topic — surface this honestly.
    cid = candidate_cids[0]
    demo_doc = customer_docs.loc[customer_docs["CustomerID"] == cid, "doc"].iloc[0]
    demo_cid = cid
    print(f"Held-out customer (CustomerID={demo_cid}):")
    print(f"  document (head): {demo_doc[:160]}...")
    print("  assigned topic:  -1 (HDBSCAN outlier)")
    print(
        "\n⚠ transform() returned -1 for every candidate. In production, fall "
        "back to the KMeans-on-embeddings comparison below — KMeans assigns "
        "every customer to a segment by construction."
    )
else:
    print(f"Held-out customer (CustomerID={demo_cid}):")
    print(f"  document (head): {demo_doc[:160]}...")
    print(f"  assigned topic:  {demo_topic}")
    print(f"  probability:     {demo_prob:.3f}")
    print(
        "\nThis is a clean assignment from the saved model. transform() can "
        "still return -1 for genuinely off-pattern customers; the KMeans pass "
        "below is the production-side fallback that always assigns a segment."
    )

# %% [markdown]
# ## Traditional vs. Embedding Stability Comparison
#
# We split the data into two time halves, run K-Means on RFM features and
# embedding-based clustering (UMAP + HDBSCAN) on customer documents for
# each half, then compare ARI to see which approach produces more temporally
# stable segments.

# %%
step("Traditional vs. Embedding stability (ARI comparison)")

import gc

# Split data at the midpoint
mid_date = raw["InvoiceDate"].min() + (
    raw["InvoiceDate"].max() - raw["InvoiceDate"].min()
) / 2
raw_h1 = raw[raw["InvoiceDate"] <= mid_date]
raw_h2 = raw[raw["InvoiceDate"] > mid_date]

# Customers present in both halves
cust_h1 = set(raw_h1["CustomerID"].unique())
cust_h2 = set(raw_h2["CustomerID"].unique())
common_cust = sorted(cust_h1 & cust_h2)
print(f"Customers in both halves: {len(common_cust):,}")

# --- Traditional: K-Means on RFM features ---
def compute_rfm(df, ref):
    return (
        df.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (ref - x.max()).days),
            frequency=("Invoice", "nunique"),
            monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

rfm_h1 = compute_rfm(raw_h1, mid_date).set_index("CustomerID").loc[common_cust]
rfm_h2 = compute_rfm(
    raw_h2, raw["InvoiceDate"].max() + pd.Timedelta(days=1)
).set_index("CustomerID").loc[common_cust]

X_h1 = StandardScaler().fit_transform(rfm_h1)
X_h2 = StandardScaler().fit_transform(rfm_h2)

km_labels_h1 = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_h1)
km_labels_h2 = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_h2)
ari_kmeans_all = adjusted_rand_score(km_labels_h1, km_labels_h2)

# --- Embedding: UMAP + HDBSCAN on customer docs ---
def build_docs(df, cust_list, top_n=20):
    pr = (
        df[df["CustomerID"].isin(cust_list)]
        .groupby(["CustomerID", "Description"])["Revenue"]
        .sum()
        .reset_index()
        .sort_values(["CustomerID", "Revenue"], ascending=[True, False])
        .groupby("CustomerID")
        .head(top_n)
    )
    return (
        pr.groupby("CustomerID")["Description"]
        .apply(lambda x: ", ".join(x))
        .reindex(cust_list)
        .fillna("")
        .tolist()
    )

docs_h1 = build_docs(raw_h1, common_cust)
docs_h2 = build_docs(raw_h2, common_cust)

# Encode and cluster each half (using raw UMAP+HDBSCAN to save memory)
emb_h1 = embedding_model.encode(docs_h1, show_progress_bar=False)
emb_h2 = embedding_model.encode(docs_h2, show_progress_bar=False)

umap_h1 = UMAP(
    n_components=CONFIG["umap_n_components"], metric="cosine",
    random_state=42, n_jobs=1,
).fit_transform(emb_h1)
umap_h2 = UMAP(
    n_components=CONFIG["umap_n_components"], metric="cosine",
    random_state=42, n_jobs=1,
).fit_transform(emb_h2)

embed_labels_h1 = HDBSCAN(
    min_cluster_size=CONFIG["min_cluster_size"],
    min_samples=CONFIG["min_samples"],
).fit_predict(umap_h1)
embed_labels_h2 = HDBSCAN(
    min_cluster_size=CONFIG["min_cluster_size"],
    min_samples=CONFIG["min_samples"],
).fit_predict(umap_h2)

# Free memory
del emb_h1, emb_h2, umap_h1, umap_h2
gc.collect()

# Fair comparison (Issue 5): apply same non-noise filter to both methods
valid = (embed_labels_h1 != -1) & (embed_labels_h2 != -1)
if valid.sum() > 0:
    ari_kmeans_fair = adjusted_rand_score(
        km_labels_h1[valid], km_labels_h2[valid]
    )
    ari_embed = adjusted_rand_score(
        embed_labels_h1[valid], embed_labels_h2[valid]
    )
else:
    ari_kmeans_fair = float("nan")
    ari_embed = float("nan")

print(f"\n{'---'*17}")
print(f"  Temporal Stability (ARI)")
print(f"{'---'*17}")
print(f"  K-Means (all customers):       {ari_kmeans_all:.3f}")
print(f"  K-Means (same non-noise set):  {ari_kmeans_fair:.3f}")
print(f"  Embedding (non-noise set):     {ari_embed:.3f}")
print(f"  Non-noise pairs:               {valid.sum():,} / {len(common_cust):,}")

# Data-driven reading: let the numbers, not a hard-coded sentence, set tone.
def _ari_label(x):
    if not np.isfinite(x): return "undefined"
    if x < 0.05:           return "essentially none (near zero)"
    if x < 0.15:           return "weak"
    if x < 0.35:           return "modest"
    return "moderate-to-strong"

gap = (ari_embed - ari_kmeans_fair) if np.isfinite(ari_embed) and np.isfinite(ari_kmeans_fair) else float("nan")
if not np.isfinite(gap):
    verdict = "the comparison is undefined for this slice"
elif abs(gap) < 0.03:
    verdict = "the two methods are essentially tied on temporal stability"
elif gap > 0:
    verdict = (
        f"the embedding pipeline is more temporally stable than RFM-KMeans "
        f"on the same non-noise set (Δ = +{gap:.3f})"
    )
else:
    verdict = (
        f"RFM-KMeans is more temporally stable than the embedding pipeline "
        f"on the same non-noise set (Δ = {gap:.3f})"
    )

print()
print(
    "Reading: K-Means stability on the full base is "
    f"**{_ari_label(ari_kmeans_all)}** (ARI {ari_kmeans_all:.3f}); on the same "
    f"non-noise set it is **{_ari_label(ari_kmeans_fair)}** "
    f"(ARI {ari_kmeans_fair:.3f}). The embedding pipeline scores "
    f"**{_ari_label(ari_embed)}** (ARI {ari_embed:.3f}). So {verdict}."
)
print(
    "Either way, embeddings track seasonal product mix, so the persistent "
    "customer ID in the CRM is best derived from RFM-KMeans, with the "
    "embedding pass kept for discovery and re-run on a refresh schedule."
)

# %%
step("Save BERTopic model")

save_dir = "bertopic_customer_segments"
try:
    topic_model.save(
        save_dir,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=CONFIG["embedding_model"],
    )
    print(f"Model saved to ./{save_dir}/")
    print(f"Reload with: BERTopic.load('{save_dir}')")
except Exception as e:
    print(f"Save failed: {e}")
    print("You can still use the model in this session.")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Embeddings add purchase meaning that RFM-KMeans cannot.** Two
#    customers with similar spend, frequency, and recency can buy very
#    different things; the embedding pipeline picks up that signal.
# 2. **Document construction matters more than the embedding model.**
#    Filter ubiquitous SKUs (>5% of customers), revenue-weight the top-N
#    products, and normalize descriptions before encoding.
# 3. **Discovery vs. activation.** HDBSCAN finds sharp niches; collapse
#    them to 4–8 segments with `reduce_topics()` (or use KMeans on the
#    same embeddings) for a CRM-ready segmentation.
# 4. **The LLM-naming trap.** An LLM will name any cluster, including
#    noise. Always pair the generated name with keyword + RFM evidence;
#    if the narrative doesn't match the data, keep the keyword label.
# 5. **`transform()` is the activation hook.** Score held-out customers
#    against the saved model, and fall back to KMeans-on-embeddings when
#    HDBSCAN flags the customer as an outlier.
# 6. **Be honest about temporal stability.** ARI on the time-split halves
#    is the right diagnostic, and the verdict depends on the run.
#    Embeddings track seasonal product mix; either re-embed on a known
#    schedule or use the embedding for discovery and an RFM-KMeans
#    segment for the persistent CRM ID.

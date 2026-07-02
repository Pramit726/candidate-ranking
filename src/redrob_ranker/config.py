"""Single source of truth for weights, paths, thresholds, and filter constants.

All tunable knobs for the Redrob ranking pipeline live here. No logic — only
configuration values imported by the rest of the package.
"""

import re

DEFAULT_DATA_DIR = "../data"
DEFAULT_MODELS_DIR = "../models"
DEFAULT_OUTPUT = "../data/final_ranking.csv"
DEFAULT_TOP_N = 100
DEFAULT_AS_OF_DATE = "2026-05-27"

# ── Test-set files only ───────────────────────────────────────────────────────
GOLDEN_SPLIT_FILES = ("golden_test.jsonl",)
GOLDEN_COMBINED = "golden_test.jsonl"
POOL_FILES = ("test_set.jsonl",)

RANKER_MODEL_FILE = "xgboost_ranker.json"

# ── XGBoost Tech_Fit classifier ───────────────────────────────────────────────
TECH_CLF_PARAMS = dict(
    objective="multi:softmax",
    num_class=5,  # overridden at runtime
    eval_metric="mlogloss",
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    verbosity=0,
)

# ── LambdaMART ranker ─────────────────────────────────────────────────────────
LTR_PARAMS = dict(
    tree_method="hist",
    objective="rank:ndcg",
    lambdarank_num_pair_per_sample=8,
    max_depth=4,
    learning_rate=0.1,
    n_estimators=150,
    random_state=42,
)

# ── Weighted scoring ──────────────────────────────────────────────────────────
W_TECH = 0.50
W_CONTEXT = 0.30
W_BEHAVIOR = 0.20
MAX_WEIGHTED_SCORE = W_TECH * 4 + W_CONTEXT * 4 + W_BEHAVIOR * 4  # 4.0

# ── Composite evaluation weights ──────────────────────────────────────────────
COMPOSITE_WEIGHTS = {"NDCG@10": 0.50, "NDCG@50": 0.30, "MAP": 0.15, "P@10": 0.05}

# ── Behavior_Fit thresholds ───────────────────────────────────────────────────
RECENCY_RECENT = 14
RECENCY_MILD = 45
RECENCY_REAL = 90
SIX_MONTHS = 180
OPEN_TO_WORK_CAP = 3

# ── Context_Fit: consulting firms & management titles ─────────────────────────
_CONSULTING_FIRMS = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "cognizant",
    "capgemini",
    "accenture",
    "mphasis",
    "hexaware",
    "tech mahindra",
    "l&t infotech",
    "hcl technologies",
    "mindtree",
    "ltimindtree",
    "persistent systems",
    "niit technologies",
    "mastech",
}
_CONSULTING_INDUSTRIES = {"consulting", "it services", "it services and consulting"}
_MGMT_TITLE_TOKENS = [
    "director",
    "vp ",
    "vice president",
    "engineering manager",
    "head of",
    "cto",
    "ceo",
]


WRONG_TITLE_KEYWORDS = [
    "marketing",
    "sales",
    "hr ",
    "human resource",
    "recruiter",
    "talent acquisition",
    "finance",
    "accountant",
    "accounting",
    "business development",
    "bd manager",
    "supply chain",
    "logistics",
    "operations manager",
    "project manager",
    "product manager",
    "graphic design",
    "ux designer",
    "ui designer",
    "visual design",
    "content writer",
    "copywriter",
    "seo",
    "social media",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "hardware engineer",
    "embedded",
    "firmware",
    "computer vision",
    "cv engineer",
    "speech",
    "audio",
    "robotics",
    "autonomous",
    "teacher",
    "professor",
    "lecturer",
    "faculty",
    "doctor",
    "physician",
    "nurse",
    "pharmacist",
    "lawyer",
    "legal",
    "compliance",
    "customer success",
    "customer support",
    "support engineer",
]

PURE_CONSULTING_FIRMS = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "mindtree",
    "ltimindtree",
    "l&t infotech",
    "persistent systems",
    "niit technologies",
    "mastech",
    "kforce",
    "syntel",
    "unison",
}

INDIA_VARIANTS = {"india", "in", "ind"}
MIN_YOE = 3
MAX_YOE = 12
MAX_INACTIVE_DAYS = 180
MIN_RESPONSE_RATE = 0.10

TITLE_LEVELS = {
    "junior": 1,
    "associate": 1,
    "intern": 1,
    "trainee": 1,
    "engineer": 2,
    "developer": 2,
    "analyst": 2,
    "scientist": 2,
    "senior": 3,
    "sr.": 3,
    "sr ": 3,
    "lead": 3,
    "staff": 4,
    "specialist": 4,
    "expert": 4,
    "principal": 5,
    "architect": 5,
    "director": 6,
    "vp": 6,
    "head": 6,
    "manager": 6,
}

DOMAIN_KEYWORD_GROUPS = {
    "core_ml": {
        "machine learning",
        "deep learning",
        "nlp",
        "natural language",
        "llm",
        "large language",
        "rag",
        "retrieval augmented",
    },
    "search_ranking": {
        "search",
        "retrieval",
        "ranking",
        "recommendation",
        "recommender",
        "information retrieval",
        "semantic search",
        "hybrid search",
    },
    "embeddings_models": {
        "embedding",
        "vector",
        "sentence-transformer",
        "bge",
        "e5",
        "bert",
        "transformer",
    },
    "vector_dbs": {
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "opensearch",
        "elasticsearch",
        "faiss",
        "annoy",
        "scann",
    },
    "evaluation": {
        "ndcg",
        "mrr",
        "a/b test",
        "offline eval",
        "online eval",
        "precision@",
        "recall@",
    },
    "production_systems": {
        "production",
        "deployed",
        "inference",
        "latency",
        "throughput",
    },
}

# Pre-compile title keyword pattern
_TITLE_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in WRONG_TITLE_KEYWORDS), re.IGNORECASE
)
# Pre-compile domain group patterns
_COMPILED_DOMAIN_GROUPS = {
    grp: re.compile(
        "|".join(
            (
                (r"\b" + re.escape(kw) + r"\b")
                if kw[-1].isalnum()
                else (r"\b" + re.escape(kw))
            )
            for kw in kws
        ),
        re.IGNORECASE,
    )
    for grp, kws in DOMAIN_KEYWORD_GROUPS.items()
}

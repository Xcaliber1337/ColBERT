from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

CHECKPOINT_PATH = "colbert-ir/colbertv2.0"
COLLECTION_PATH = "/data/collection.tsv"
INDEX_ROOT = "/data/experiments"
INDEX_NAME = "myindex.nbits=2"

config = ColBERTConfig(
    doc_maxlen=180,
    dim=128,
    nbits=2,
    nway=4,
    similarity='cosine'
)

def run_index():
    # FIX: Only pass RunConfig here
    with Run().context(RunConfig(nranks=1, experiment="mvp", index_root=INDEX_ROOT)):
        indexer = Indexer(checkpoint=CHECKPOINT_PATH, config=config)
        indexer.index(name=INDEX_NAME, collection=COLLECTION_PATH, overwrite=True)

if __name__ == "__main__":
    run_index()

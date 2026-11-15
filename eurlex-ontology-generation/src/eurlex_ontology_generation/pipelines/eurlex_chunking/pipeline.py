from kedro.pipeline import Pipeline, node

from .nodes import chunk_corpus


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=chunk_corpus,
                inputs=[
                    "eurlex_corpus",
                    "params:eurlex_chunking.text_column",
                    "params:eurlex_chunking.chunk_size",
                    "params:eurlex_chunking.chunk_overlap",
                    "params:eurlex_chunking.strategy",
                    "params:eurlex_chunking.include_preamble",
                    "params:eurlex_chunking.include_annexes",
                ],
                outputs="eurlex_full_chunks",
                name="chunk_full_corpus_node",
            ),
        ]
    )

from kedro.pipeline import Pipeline, node
from .nodes import prepare_corpus


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_corpus,
                inputs=[
                    "eurlex_raw",
                    "params:selected_columns",
                    "params:text_column",
                    "params:drop_missing_text",
                ],
                outputs="eurlex_corpus",
                name="prepare_corpus_node",
            ),
        ]
    )
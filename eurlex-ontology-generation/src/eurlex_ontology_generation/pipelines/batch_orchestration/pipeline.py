from kedro.pipeline import Pipeline, node

from .nodes import update_batch_status


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=update_batch_status,
            inputs=[
                "chunk_batch_manifest",
                "params:batch_orchestration.batch_id",
                "params:batch_orchestration.status",
                "params:batch_orchestration.manifest_path",
            ],
            outputs="updated_chunk_batch_manifest",
            name="update_batch_status_node",
        ),
    ])
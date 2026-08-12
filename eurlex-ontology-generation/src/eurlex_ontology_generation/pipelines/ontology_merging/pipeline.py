from kedro.pipeline import Pipeline, node
from .nodes import (
    parse_ontology_candidates,
    review_intra_ontology_concepts,
    iterative_merge_ontologies,
)


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            node(
                func=parse_ontology_candidates,
                inputs="ontology_candidates",
                outputs="parsed_ontologies",
                name="parse_ontology_candidates_node",
            ),
            node(
                func=review_intra_ontology_concepts,
                inputs=[
                    "parsed_ontologies",
                    "params:ontology_merging.model",
                    "params:ontology_merging.temperature",
                    "params:ontology_merging.intra_review_system_prompt",
                    "params:ontology_merging.intra_review_user_prompt_template",
                    "params:ontology_merging.json_mode",
                ],
                outputs=["reviewed_ontologies", "intra_review_report"],
                name="review_intra_ontology_concepts_node",
            ),
            node(
                func=iterative_merge_ontologies,
                inputs=[
                    "reviewed_ontologies",
                    "params:ontology_merging.model",
                    "params:ontology_merging.temperature",
                    "params:ontology_merging.merge_system_prompt",
                    "params:ontology_merging.merge_user_prompt_template",
                    "params:ontology_merging.merge_judge_system_prompt",
                    "params:ontology_merging.merge_judge_user_prompt_template",
                    "params:ontology_merging.json_mode",
                ],
                outputs=["final_merged_ontology", "merge_trace"],
                name="iterative_merge_ontologies_node",
            ),
        ]
    )
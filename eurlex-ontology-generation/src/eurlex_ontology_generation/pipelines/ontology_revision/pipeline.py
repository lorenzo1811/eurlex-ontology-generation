
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import revise_ontologies, validate_revised_ontologies


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=revise_ontologies,
                inputs=[
                    "ontology_reviews",
                    "params:ontology_revision.max_candidates",
                    "params:ontology_revision.model",
                    "params:ontology_revision.temperature",
                    "params:ontology_revision.system_prompt",
                    "params:ontology_revision.user_prompt_template",
                    "params:ontology_revision.json_mode",
                    "params:ontology_revision.expected_ontology_keys",
                    "params:ontology_revision.num_ctx",
                    "params:ontology_revision.num_predict",
                ],
                outputs="ontology_revised",
                name="revise_ontologies_node",
            ),
            node(
                func=validate_revised_ontologies,
                inputs=[
                    "ontology_revised",
                    "params:ontology_revision.expected_ontology_keys",
                    "params:ontology_revision.disallowed_currency_terms",
                    "params:ontology_revision.valid_odrl_core_types",
                    "params:ontology_revision.currency_check_exclusions",
                ],
                outputs="ontology_revision_report",
                name="validate_revised_ontologies_node",
            ),
        ]
    )
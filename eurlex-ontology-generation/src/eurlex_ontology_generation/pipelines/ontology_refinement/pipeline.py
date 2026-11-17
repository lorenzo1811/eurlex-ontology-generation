from kedro.pipeline import Pipeline, node

from .nodes import (
    load_concept_links,
    load_curation,
    load_merged_ontology,
    refine_ontology,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_merged_ontology,
                inputs="params:ontology_refinement.input_path",
                outputs="merged_ontology_for_refinement",
                name="load_merged_ontology_node",
            ),
            node(
                func=load_curation,
                inputs="params:ontology_refinement.curation_path",
                outputs="accepted_curation_decisions",
                name="load_curation_node",
            ),
            node(
                func=load_concept_links,
                inputs="params:ontology_refinement.links_path",
                outputs="concept_links",
                name="load_concept_links_node",
            ),
            node(
                func=refine_ontology,
                inputs=[
                    "merged_ontology_for_refinement",
                    "params:ontology_refinement.wikidata",
                    "params:ontology_refinement.vocabulary_path",
                    "accepted_curation_decisions",
                    "params:ontology_refinement.min_documents_for_class",
                    "concept_links",
                ],
                # both are catalog datasets (conf/base/catalog.yml)
                outputs=["refined_ontology", "refinement_report"],
                name="refine_ontology_node",
            ),
        ]
    )

import pandas as pd
from ollama import chat


# Generate a candidate ontology for each chunk using a local LLM (Ollama)
def generate_partial_ontologies(
    chunks: pd.DataFrame,
    max_chunks: int,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
) -> pd.DataFrame:

    # Store one ontology candidate for each processed chunk
    results = []

    # Process only the first max_chunks
    for _, row in chunks.head(max_chunks).iterrows():

        # Build the user prompt by inserting the chunk text
        user_prompt = user_prompt_template.format(
            text=row["chunk_text"]
        )

        # Query the local LLM through Ollama
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": temperature,
            },
        )

        # Extract the textual response returned by the model
        ontology_text = response["message"]["content"]

        # Save both the generated ontology and useful metadata
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "CELEX": row["CELEX"],
                "ontology": ontology_text,
            }
        )

    return pd.DataFrame(results)
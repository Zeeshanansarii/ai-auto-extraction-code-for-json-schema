import numpy as np
import json
import openai
import logging
from sentence_transformers import SentenceTransformer
from jsonschema import validate, ValidationError
from typing import Dict, Tuple, List, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm

# Configuration
openai.api_key = "Put OpenAI api key over here to execute in your system"  # OpenAI API key
MODEL = "gpt-4"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
MAX_WORKERS = 10
CHUNK_SIZE = 500  # Word count for text chunks
TOP_K_CHUNKS = 5  # Number of relevant chunks to retrieve

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@lru_cache(maxsize=100)
def cached_embed_text(text: str) -> np.ndarray:
    return EMBEDDING_MODEL.encode([text], convert_to_numpy=True)[0]

#Parse JSON schema to extract information.
def parse_schema(schema: Dict) -> List[Dict]:
    fields = []
    stack = [(schema, "")]
    while stack:
        current, path = stack.pop()
        if "properties" in current:
            for key, value in current["properties"].items():
                new_path = f"{path}.{key}" if path else key
                if value.get("type") == "object":
                    stack.append((value, new_path))
                elif value.get("type") == "array" and "items" in value:
                    if value["items"].get("type") == "object":
                        stack.append((value["items"], f"{new_path}[]"))
                    else:
                        fields.append({"path": f"{new_path}[]", **value})
                else:
                    fields.append({"path": new_path, **value})
        if "patternProperties" in current:
            for pattern, prop_schema in current["patternProperties"].items():
                fields.append({"path": f"{path}.*{pattern}", **prop_schema})
    return fields

# Generate text chunks from the input text.
def split_text_generator(text: str, chunk_size: int = CHUNK_SIZE) -> Generator[str, None, None]:
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Embed text chunks using SentenceTransformer.
def embed_text_chunks(chunks: List[str]) -> np.ndarray:
    return EMBEDDING_MODEL.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

#Retrieve top-k relevant text chunks for a field using similarity search.
def retrieve_relevant_chunks(field_desc: str, chunks: List[str], embedding: np.ndarray, top_k: int = TOP_K_CHUNKS) -> List[str]:
    desc_embedding = cached_embed_text(field_desc)
    similarities = np.dot(embedding, desc_embedding) / (np.linalg.norm(embedding, axis=1) * np.linalg.norm(desc_embedding))
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]

def extract_value(field: Dict, relevant_chunks: List[str]) -> Tuple[Any, str]:
    prompt = (
        f"Extract the value for the field described as '{field.get('description', field['path'])}' "
        f"from the following text:\n\n{' '.join(relevant_chunks)}\n\n"
        f"The value should be a {field['type']}."
    )
    if "enum" in field:
        prompt += f" Choose from: {', '.join(field['enum'])}."
    prompt += (
        "\nProvide a confidence level (high, medium, low) based on clarity in the text. "
        "Output as JSON: {'value': ..., 'confidence': ...}\n\n"
        "Confidence levels:\n- high: explicitly stated\n- medium: inferred\n- low: guessed or not found"
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result["value"], result["confidence"]
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.error(f"Error extracting value for {field['path']}: {e}")
        return None, "low"

# Group related fields to optimize large language model calls.
def group_fields(fields: List[Dict]) -> List[List[Dict]]:
    groups = {}
    for field in fields:
        parent = '.'.join(field["path"].split('.')[:-1]) or "root"
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(field)
    return list(groups.values())

# Extract values in parallel using multithreading.
def parallel_extract(fields: List[Dict], chunks: List[str], embeddings: np.ndarray) -> Tuple[Dict, Dict]:
    values = {}
    confidences = {}

    def process_field(field: Dict) -> Tuple[str, Any, str]:
        relevant_chunks = retrieve_relevant_chunks(field.get("description", field["path"]), chunks, embeddings)
        value, confidence = extract_value(field, relevant_chunks)
        return field["path"], value, confidence

    with ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
        future_to_field = {
            executor.submit(process_field, field): field
            for field in fields
        }
        for future in tqdm(as_completed(future_to_field), total=len(fields), desc="Extracting fields"):
            try:
                path, value, confidence = future.result()
                values[path] = value
                confidences[path] = confidence
            except Exception as e:
                field = future_to_field[future]
                logger.error(f"Error processing {field['path']}: {e}")
                values[field["path"]] = None
                confidences[field["path"]] = "low"

    return values, confidences

# Assemble extracted values into JSON structure matching the schema.
def assemble_json(fields: List[Dict], values: Dict, schema: Dict) -> Dict:
    result = {}
    array_indices = {}
    for field in fields:
        path_parts = field["path"].split('.')
        current = result
        for i, part in enumerate(path_parts):
            if part.endswith('[]'):
                part = part[:-2]
                if part not in current:
                    current[part] = []
                if i == len(path_parts) - 1:
                    if field["path"] not in array_indices:
                        array_indices[field["path"]] = 0
                    else:
                        array_indices[field["path"]] += 1
                    index = array_indices[field["path"]]
                    while len(current[part]) <= index:
                        current[part].append({})
                    if values[field["path"]] is not None:
                        current[part][index] = values[field["path"]]
            else:
                if part not in current:
                    current[part] = {}
                if i == len(path_parts) - 1:
                    current[part] = values[field["path"]]
                else:
                    current = current[part]

    try:
        validate(instance=result, schema=schema)
        logger.info("JSON output validated successfully")
    except ValidationError as e:
        logger.error(f"Schema validation error: {e}")
    return result

#  Convert unstructured text to structured JSON based on a schema.
def text_to_json(text: str, schema: Dict) -> Tuple[Dict, Dict]:
    if not text or not schema:
        logger.error("Empty text or schema provided")
        return {}, {}

    # Validate schema
    if schema.get("type") != "object":
        raise ValueError("Schema must be an object")

    # Parse schema
    fields = parse_schema(schema)
    logger.info(f"Parsed {len(fields)} fields from schema")

    # Preprocess text
    chunks = list(split_text_generator(text))
    logger.info(f"Split text into {len(chunks)} chunks")
    if not chunks:
        logger.error("No text chunks generated")
        return {}, {}

    embeddings = embed_text_chunks(chunks)
    values = {}
    confidences = {}
    field_groups = group_fields(fields) if len(fields) < 50 else [[f] for f in fields]

    for group in tqdm(field_groups, desc="Processing field groups"):
        if len(group) > 1 and not any(f["path"].endswith('[]') for f in group):
            descs = [f.get("description", f["path"]) for f in group]
            schema_desc = json.dumps({f["path"]: {k: v for k, v in f.items() if k != "path"} for f in group}, indent=2)
            relevant_chunks = set()
            for field in group:
                relevant_chunks.update(retrieve_relevant_chunks(field.get("description", field["path"]), chunks, embeddings))
            relevant_chunks = list(relevant_chunks)[:TOP_K_CHUNKS]

            prompt = (
                f"Extract the following fields from the text:\n{schema_desc}\n\n"
                f"Text:\n{' '.join(relevant_chunks)}\n\n"
                f"Provide extracted values with confidence levels (high, medium, low) in JSON format: "
                f"{{'values': {{field: value, ...}}, 'confidences': {{field: confidence, ...}}}}\n\n"
                f"Confidence levels:\n- high: explicitly stated\n- medium: inferred\n- low: guessed or not found"
            )
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0
                )
                result = json.loads(response.choices[0].message.content)
                for field, value in result["values"].items():
                    values[field] = value
                    confidences[field] = result["confidences"][field]
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error extracting group: {e}")
                for field in group:
                    values[field["path"]] = None
                    confidences[field["path"]] = "low"
        else:
            group_values, group_confidences = parallel_extract(group, chunks, embeddings)
            values.update(group_values)
            confidences.update(group_confidences)

    json_output = assemble_json(fields, values, schema)
    logger.info("JSON output assembled")
    return json_output, confidences

if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "project": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The project name"},
                    "budget": {"type": "number", "description": "The project budget in USD"},
                    "manager": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Manager's name"},
                            "email": {"type": "string", "description": "Manager's email"}
                        }
                    }
                }
            },
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "desc": {"type": "string", "description": "Requirement description"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "Priority level"}
                    }
                }
            },
            "status": {"type": "string", "enum": ["active", "inactive"], "description": "Project status"}
        }
    }

    prototype_text = """The project, named "IndiaGPT," has a budget of 50000 USD and is managed by Wow (wow@indiagpt.com). 
    Requirements include: 1) Fast delivery, which is high priority; 2) Cost efficiency, medium priority. 
    The project is currently active."""

    json_result, confidence_map = text_to_json(prototype_text, schema)
    print("JSON Output:", json.dumps(json_result, indent=2))
    print("Confidence Map:", json.dumps(confidence_map, indent=2))
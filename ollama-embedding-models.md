# Embedding models · Ollama Blog
[![Embedding models](https://ollama.com/public/blog/embedding-models.png)](https://ollama.com/download)

Ollama supports embedding models, making it possible to build retrieval augmented generation (RAG) applications that combine text prompts with existing documents or other data.

What are embedding models?
--------------------------

Embedding models are models that are trained specifically to generate _vector embeddings_: long arrays of numbers that represent semantic meaning for a given sequence of text:

![what-are-embedding-models](https://ollama.com/public/blog/what-are-embeddings.svg)

The resulting vector embedding arrays can then be stored in a database, which will compare them as a way to search for data that is similar in meaning.

Example embedding models
------------------------


|Model            |Parameter Size|          |
|-----------------|--------------|----------|
|mxbai-embed-large|334M          |View model|
|nomic-embed-text |137M          |View model|
|all-minilm       |23M           |View model|


Usage
-----

To generate vector embeddings, first pull a model:

```
ollama pull mxbai-embed-large

```


Next, use the [REST API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings), [Python](https://github.com/ollama/ollama-python) or [JavaScript](https://github.com/ollama/ollama-js) libraries to generate vector embeddings from the model:

REST API

```
curl http://localhost:11434/api/embeddings -d '{
  "model": "mxbai-embed-large",
  "prompt": "Llamas are members of the camelid family"
}'

```


Python library

```
ollama.embeddings(
  model='mxbai-embed-large',
  prompt='Llamas are members of the camelid family',
)

```


Javascript library

```
ollama.embeddings({
    model: 'mxbai-embed-large',
    prompt: 'Llamas are members of the camelid family',
})

```


Ollama also integrates with popular tooling to support embeddings workflows such as [LangChain](https://python.langchain.com/docs/integrations/text_embedding/ollama/) and [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/).

Example
-------

This example walks through building a retrieval augmented generation (RAG) application using Ollama and embedding models.

### Step 1: Generate embeddings

```
pip install ollama chromadb

```


Create a file named `example.py` with the contents:

```
import ollama
import chromadb

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

```


### Step 2: Retrieve

Next, add the code to retrieve the most relevant document given an example prompt:

```
# an example prompt
prompt = "What animals are llamas related to?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

```


### Step 3: Generate

Lastly, use the prompt and the document retrieved in the previous step to generate an answer!

```
# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama2",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])

```


Then, run the code:

```
python example.py

```


Llama 2 will answer the prompt `What animals are llamas related to?` using the data:

```
Llamas are members of the camelid family, which means they are closely related to two other animals: vicuñas and camels. All three species belong to the same evolutionary lineage and share many similarities in terms of their physical characteristics, behavior, and genetic makeup. Specifically, llamas are most closely related to vicuñas, with which they share a common ancestor that lived around 20-30 million years ago. Both llamas and vicuñas are members of the family Camelidae, while camels belong to a different family (Dromedary).

```


### Coming soon

More features are coming to support workflows that involve embeddings:

*   **Batch embeddings:** processing multiple input data prompts simultaneously
*   **OpenAI API Compatibility**: support for the `/v1/embeddings` OpenAI-compatible endpoint
*   **More embedding model architectures:** support for ColBERT, RoBERTa, and other embedding model architectures

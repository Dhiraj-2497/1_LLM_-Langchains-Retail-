# 1_LLM_-Langchains-Retail-
Data stored in database often requires programming knowledge to be accessed. Here, we create a MYSQL database and use LLM, Langchain, Vector Database to build a tool such that data can be extracted using Layman language and we have used Streamlit to create a simple UI

# LLM Chains - query generator
Using LLM chains to generate result from SQL database using prompts.

<b> Steps </b>:
1. Using Google's Gen AI module to use LLM model
2. Created a local database in mysql
3. Connected to the local database using Langchain's SQLDatabase module
4. Used HuggingFace's Embeddings to create embeddings
5. Chromadb from langchain vectores to store embeddings
6. Semantic similarity selection from langchain to determine similar examples
7. Few shots prompt from Langchians to determine Prompts
8. Finally used Streamlit to host it

<b> Modules / Tools used </b>:
* Google's GenAI, MYSQL, Langchain's SQLDatabase module, HuggingFace's Embeddings, Chromadb, Semantic similarity, Langchain's Few shots prompt, STreamlit

We first created a raw code/product in Jupyter notebook as we can see in HTML file and then created a modularised package.

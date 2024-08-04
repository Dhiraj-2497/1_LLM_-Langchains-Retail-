from langchain_google_genai import GoogleGenerativeAI
import urllib
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
import os

# from dotenv import load_dotenv
# load_dotenv()
from few_shots_examples import few_shots

api_key = 'add your api key'

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key,
                         temperature=0.1)


def get_query_chain():
    #creating db pbject
    db_user = 'root'
    db_password = urllib.parse.quote_plus('Password')
    db_host = 'localhost'
    db_name = 'atliq_tshirts'

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [' '.join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(texts=to_vectorize, embedding=embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=3,
    )

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  #These variables are used in the prefix and suffix
    )

    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return new_chain


if __name__ == "__main__":
    chain = get_query_chain()
    chain.run("How many total T shirts do we have in stock?")

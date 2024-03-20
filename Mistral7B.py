
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# Connect CSV to DB
database_file = "database_LLM.db"  # Replace with your database filename
df = pd.read_csv("Data/LLM_data.csv")  # Replace with your CSV filename
df['DemandDate'] = pd.to_datetime(df['DemandDate'], format='%m/%d/%y')
conn = sqlite3.connect(database_file)
df.to_sql("LLM_sql", conn, if_exists="replace", index=False)
conn.close()

def load_quantized_model(model_id, quantization_config):
    """Loads a quantized Hugging Face Transformer model.

    Args:
        model_id (str): The model identifier on the Hugging Face Hub.
        quantization_config (BitsAndBytesConfig): Configuration for quantization.

    Returns:
        AutoModelForCausalLM: The loaded and quantized model.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=quantization_config
    )
    return model

def create_text_generation_pipeline(model, tokenizer):
    """Creates a text generation pipeline with a quantized model.

    Args:
        model (AutoModelForCausalLM): The quantized language model.
        tokenizer (AutoTokenizer): The corresponding tokenizer.

    Returns:
        pipeline: The configured text generation pipeline.
    """

    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipeline

def generate_sqlite_query(question,pipeline):
    """Generates a SQLite query using an LLMChain and prompt.

    Args:
        question (str): The natural language question for query generation and the model pipeline

    Returns:
        str: The generated SQLite query.
    """
    llm = HuggingFacePipeline(pipeline=pipeline)
    ####### Prompt template definition ######
    template="""<s>[INST] You are a helpful, respectful and honest SQLlite assistant. Return the appropriate SQLlite query given the table and its respective schema as below:

table name : LLM_sql
Column names : PartID, RepairID, CarID, DemandDate, Quantity, ModelYear.
Here is the description of each column:
PartID: Unique identifier for each part.(Character)
RepairID: Unique identifier for each repair instance. It has only one record (Alphanumeric)
CarID: Unique identifier for cars; .(Alphanumeric)
DemandDate: Date of repair demand ((datetime) format)
Quantity: Number of parts used or required for the repair.(Integer)
ModelYear: Year the car model was manufactured.(Integer)


For the given question: {question}
Follow the following guidelines:
1.If the question mentions just year but not model year, then strictly consider extracting  information from DemandDate.
2.If not explicitly mentioned, write the sqlite query for fetching all the rows, dont use limit.
3.Opt for a easier route to handle the question than going for subqueries
4. Since it is sqlite, year(),month(),day() methods are not to be used, use strftime function
Write the SQLite3 query and dont give any other description: [/INST] </s>
"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt,llm=llm)
    response = llm_chain.run({"question": question})

    return response

def query_table(sql_response):
    """Generates tables and plots basis the SQL query executed on the data

        Args:
            question (str): The extracted SQLite query

        Returns:
            str: Graphs or plots
        """
    check_words = ['plots', 'trends','plot','trend','graphs', 'graph']

    conn = sqlite3.connect('database_LLM.db')
    cursor = conn.cursor()
    df = pd.read_sql_query(sql_response, conn)

    if any(sub in question for sub in check_words):
    # Plotting
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o')  # 'o' for markers at data points
        plt.xlabel(df.columns[0])  # Assigning column name as x-axis label
        plt.ylabel(df.columns[1])  # Assigning column name as y-axis label
        plt.title('Plot Title')
        plt.grid(True)  # Add grid lines
        plt.show()

    conn.close()
    return df

# ***** Main Execution *****
if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    model = load_quantized_model(model_id, quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipeline = create_text_generation_pipeline(model, tokenizer)

    # Example usage:
    user_question = "How many cars have undergone repair in the last 10 days?"
    query = generate_sqlite_query(user_question,pipeline)
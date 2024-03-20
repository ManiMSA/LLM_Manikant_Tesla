**Introduction**

This project aims to build a Language Model (LM) capable of interacting with CSV data to derive insights using natural language input. The project utilizes OpenAI's GPT (Generative Pre-trained Transformer) models for both the OpenAI track and Mistral 7B models for the open-source track. The LM is designed to process natural language queries and generate appropriate responses in the form of graphs or tables based on the insights derived from the provided CSV data.

**Components**

  i. Code
The codebase consists of Python scripts responsible for the following functionalities:
Data preprocessing: Scripts to clean and preprocess the CSV data for compatibility with the LLM. Transform the CSV data into SQLite database.
LLM run : Scripts to run and prompt engineer the GPT models for both the OpenAI track and Mistral 7B models.
          Mistral7B.py - Mistral and OpenAI.ipynb - OpenAI.


  ii. Requirements
The requirements.txt file lists all the dependencies required to run the code. Users need to install these dependencies before executing the scripts.



**Usage**

To run the project, follow these steps:

Clone the repository to your local machine.
Install the required dependencies using pip install -r requirements.txt.
Preprocess the CSV data using the provided scripts.
Run the LLM models using the preprocessed data.
Use the ipynb files to interact with the trained models and obtain insights from the CSV data.











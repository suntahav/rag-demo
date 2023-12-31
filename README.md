# rag-demo
This is a RAG application having chat with PDF capability.

## Setup
use the db_init.py script to create the database and populate it Cohere embeddings for the text chunks in the pdf data

## AWS Integration
The lambda function for AWS Lambda is in the lambda.py file. It is simple function that extracts the user prompt 
and passes it to cohere llm along with all retrieved docs from pinecone based on similarity to the prompt. The 
function returns the cohere llm response to the user. It is integrated with the AWS API Gateway and can be accessed
by sending a post request to the API endpoint.

## Testing
Use the test notebook to test the application. Just follow the instructions in the notebook to test the application.
It already has some sample prompts and outputs.


## SK RAG ChatBot

A chat application based on an LLM grounded using RAG.

##### Required Azure resources:

1. Azure Open AI resource with `gpt-4o` and `text-embedding-ada-002` deployments.
2. Azure Cosmos DB for MongoDB (vCore) free tier cluster.


##### To run this repository locally, follow these steps:

1. Clone the repository to your local machine.

2. Install the required Python packages: `semantic-kernel`

3. Create a `.env` file at the root of the repo and copy the contents of the `.env.example`. Create the respective Azure resources and configure the `.env` file with the relavant details.

4. Start the program by running, `python main.py`

##### To add custom data:

1. Add the custom JSON file in the `./src/data.json`. Be careful with the schema as it should adhere to the one used in the code to use the correct data to generate the embeddings.
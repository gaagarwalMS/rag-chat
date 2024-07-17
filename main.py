import os  
import asyncio
import json
import time

from dotenv import load_dotenv  
from semantic_kernel import Kernel  
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding
)  

from semantic_kernel.connectors.memory.azure_cosmosdb import (
    AzureCosmosDBMemoryStore,
)
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from utils.urlEncoder import getDPConnectionString

from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.memory_store_base import MemoryStoreBase

from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings


from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable

from semantic_kernel.functions import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (  
    AzureChatPromptExecutionSettings,  
)

azure_cosmos_container = os.getenv("AZCOSMOS_CONTAINER_NAME")

async def upsert_data_to_memory_store(memory: SemanticTextMemory, store: MemoryStoreBase, data_file_path: str) -> None:
    """
    This asynchronous function takes two memory stores and a data file path as arguments.
    It is designed to upsert (update or insert) data into the memory stores from the data file.

    Args:
        memory (callable): A callable object that represents the semantic kernel memory.
        store (callable): A callable object that represents the memory store where data will be upserted.
        data_file_path (str): The path to the data file that contains the data to be upserted.

    Returns:
        None. The function performs an operation that modifies the memory stores in-place.
    """
    with open(file=data_file_path, mode="r", encoding="utf-8") as f:
        print("Loading data from file...")
        data = json.load(f)
        n = 0
        for item in data:
            n += 1
            # check if the item already exists in the memory store
            # if the id doesn't exist, it throws an exception
            try:
                already_created = bool(await store.get(azure_cosmos_container, item["id"], with_embedding=True))
            except Exception:
                already_created = False
            # if the record doesn't exist, we generate embeddings and save it to the database
            if not already_created:
                print("calling save_information...")
                await memory.save_information(
                    collection="sk-rag-container",
                    id=item["id"],
                    # the embedding is generated from the text field
                    text=item["content"],
                    description=item["title"],
                )
                print(
                    "Generating embeddings and saving new item:",
                    n,
                    "/",
                    len(data),
                    end="\r",
                )
            else:
                print("Skipping item already exits:", n, "/", len(data), end="\r")

async def main():  
    try:  
        # Get configuration settings  
        load_dotenv()
        azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")  
        azure_oai_key = os.getenv("AZURE_OAI_KEY")  
        azure_oai_gpt4o_deployment = os.getenv("AZURE_OAI_GPT4O_DEPLOYMENT")  
        azure_oai_embedding_dep = os.getenv("AZURE_OAI_EMBEDDING_DEPLOYMENT")
        azure_chat_dep_baseurl = os.getenv("AZURE_CHAT_DEP_BASEURL")

        azure_cosmos_dbname = os.getenv("AZCOSMOS_DATABASE_NAME")
        azure_cosmos_container = os.getenv("AZCOSMOS_CONTAINER_NAME")

        # Azure Cosmos DB Memory Store configuration
  
        kernel = Kernel()

        kernel.add_service(AzureChatCompletion(
            service_id="chat_completion",  
            deployment_name=azure_oai_gpt4o_deployment,  
            api_key=azure_oai_key,  
            base_url=azure_chat_dep_baseurl  
        ))
        print("Added Azure OpenAI Chat Service...\n")

        kernel.add_service(AzureTextEmbedding(
            service_id="text_embedding",
            deployment_name=azure_oai_embedding_dep,
            api_key=azure_oai_key,
            endpoint=azure_oai_endpoint
        ))
        print("Added Azure OpenAI Embedding Generation Service... \n")
      
        # Vector search index parameters
        index_name = "VectorSearchIndex"
        vector_dimensions = 1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
        num_lists = 1
        similarity = "COS"  # cosine distance

        print("Creating or updating Azure Cosmos DB Memory Store...\n")
        # create azure cosmos db for mongo db vcore api store and collection with vector ivf
        # currently, semantic kernel only supports the ivf vector kind
        store = await AzureCosmosDBMemoryStore.create(
            cosmos_connstr=getDPConnectionString(),
            cosmos_api="mongo-vcore",
            database_name=azure_cosmos_dbname,
            collection_name=azure_cosmos_container,
            index_name=index_name,
            vector_dimensions=vector_dimensions,
            num_lists=num_lists,
            similarity=similarity,
        )
        print("Finished updating Azure Cosmos DB Memory Store...\n")

        memory = SemanticTextMemory(storage=store, embeddings_generator=kernel.get_service("text_embedding"))
        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACDB")
        print("Registered Azure Cosmos DB Memory Store...\n")

        print("Upserting data to Azure Cosmos DB Memory Store...\n")
        await upsert_data_to_memory_store(memory, store, "./src/data.json")

        # each time it calls the embedding model to generate embeddings from your query
        # query_term = "What do you know about the godfather?"
        # result = await memory.search(azure_cosmos_container, query_term)

        # print(
        #     f"Result is: {result[0].text}\nRelevance Score: {result[0].relevance}\nFull Record: {result[0].additional_metadata}"
        # )

        prompt = """
        You are a chatbot that can have a conversations about any topic related to the provided context.
        Give answers relavant to the context supplemented with you knowledge. If you find the query topic not relavant to the context say, "I don't know"
        provided context: {{$db_record}}

        User: {{$query_term}}
        Chatbot:"""

        execution_settings = AzureChatPromptExecutionSettings(
            service_id="chat_completion", ai_model_id=azure_oai_gpt4o_deployment, max_tokens=500, temperature=0.0, top_p=0.5
        )

        chat_prompt_template_config = PromptTemplateConfig(
            template=prompt,
            name="grounded_response",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="db_record", description="The database record", is_required=True),
                InputVariable(name="query_term", description="The user input", is_required=True),
            ],
            execution_settings=execution_settings,
        )

        chat_function = kernel.add_function(
            function_name="ChatGPTFunc", plugin_name="chatGPTPlugin", prompt_template_config=chat_prompt_template_config
        )

        # completions_result = await kernel.invoke(
        #     chat_function, KernelArguments(query_term=query_term, db_record=result[0].additional_metadata)
        # )

        # print(completions_result)

        query_term = ""
        search_result = ""
        completions_result = ""

        while query_term != "exit":
            print("\n")
            query_term = input("Enter a query: ")
            search_result = await memory.search(azure_cosmos_container, query_term)
            completions_result = kernel.invoke_stream(
                chat_function, KernelArguments(query_term=query_term, db_record=search_result[0].additional_metadata)
            )
            print(f"\nResponse:")
            async for completion in completions_result:
                print(str(completion[0]), end="")
            print("\n")
            time.sleep(5)
  
    except Exception as ex:  
        print(f"An error occurred in the main function: {ex}")
  
if __name__ == '__main__':  
    asyncio.run(main())
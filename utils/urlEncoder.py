import os
from urllib.parse import quote_plus

def getDPConnectionString():
    # Original credentials  
    username = os.getenv("AZCOSMOS_USERNAME")
    password = os.getenv("AZCOSMOS_PASSWORD")
    mongo_resource_name = os.getenv("AZCOSMOS_RESOURCE_NAME")
    
    # URL-encode the credentials  
    encoded_username = quote_plus(username)  
    encoded_password = quote_plus(password)  
    
    # Construct the connection string  
    connection_string = f"mongodb+srv://{encoded_username}:{encoded_password}@{mongo_resource_name}.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"

    print(connection_string)

    return connection_string  


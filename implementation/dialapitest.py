import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(override=True)
dial_api_key = os.getenv('DIAL_API_KEY')

azure_model =  os.getenv('AZURE_MODEL', "gpt-4")


# You can create os environ variables: `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION` and `AZURE_OPENAI_ENDPOINT` respectively and you will not be required to pass any parameters:
client = AzureOpenAI(
    api_key         = dial_api_key,
    api_version     = "2024-08-01-preview",
    azure_endpoint  = "https://ai-proxy.lab.epam.com",
)

models = [model.id for model in client.models.list()]
#print(models)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What is the capital of Japan?"
    }
]

response = client.chat.completions.create(
  model=azure_model,
  messages=messages,
)
print(response.choices[0].message.content)
# Used to securely prompt the user for input (e.g., API key)
import getpass
# Provides a way to interact with the operating system environment
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Import OpenAI's embedding model wrapper
from langchain_openai import OpenAIEmbeddings

# Enable LangChain's V2 tracing to monitor and debug LLM chains and tools
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt the user to securely input their OpenAI API key (won't be visible as typed)
os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Create a list of Document objects, each representing a piece of content with associated metadata
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="LangChain helps building LLM-based applications",
        metadata={"source": "lang-chain-doc"},
    ),
    Document(
        page_content="LLMs have changed the scene in AI and ML",
        metadata={"source": "llm-doc"},
    ),
]

#This code takes a list of documents, turns their text into vectors (numeric representations of meaning), stores those vectors in Chroma,

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

#search for the two documents most similar in meaning to "cat".
results = vectorstore.similarity_search("cat", 2)

print('\n')

for doc in results:
  print(doc.page_content, doc)
  print("=================\n")
print('\n')

results = vectorstore.similarity_search_with_relevance_scores("cats are cool")
print('\n')
print("++++Result for Similarity Search: ", str(results))
print('\n')

# Loop through the results, which are (document, score) tuples
for doc, score in results:
  print(score) # Print the relevance score (higher = more relevant)
  print(doc.page_content) # Print the content of the matched document
  print("=================\n") # Separator for readability
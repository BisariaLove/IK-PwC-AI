import getpass
# Importing the os module to interact with the operating system and access environment variables.
import os
# Importing ChatPromptTemplate to structure and manage prompts for chat-based language models.
from langchain_core.prompts import ChatPromptTemplate
# Importing StrOutputParser to convert language model responses into plain text strings.
from langchain_core.output_parsers import StrOutputParser
# Importing ChatOpenAI to integrate and interact with OpenAI's chat models using LangChain.
from langchain_openai import ChatOpenAI
# Importing create_retrieval_chain to build retrieval-augmented question-answering chains.
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_openai_functions_agent
from langchain_classic.agents import AgentExecutor
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["OPENAI_API_KEY"] = getpass.getpass()

loader = WebBaseLoader("https://zerodha.com/")

docs = loader.load()

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# create_stuff_documents_chain :
#  Create a chain for passing a list of Documents to a model.

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""", output_parser = output_parser)

document_chain = create_stuff_documents_chain(llm, prompt) # prompt | llm


retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

retrieval_chain.invoke({"input": "Can Zerodha help in stock marketing only or in mutual fund investment too?"})


retriever_tool = create_retriever_tool(
    retriever,
    "zerodha_search",
    "Search for information about Zerodha. For any questions about Zerodha, you must use this tool!",
)


os.environ["TAVILY_API_KEY"] = getpass.getpass()

search = TavilySearchResults()

print(search.description)
print(search.name)

tools = [retriever_tool, search]

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Use zerodha_search for Zerodha questions "
        "and web search for current information. If the tools don't provide "
        "enough evidence, say you don't know.",
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

print("\n\n###Prompt created,", str(prompt))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("\n###Agent Executor Created: ")

result = agent_executor.invoke({"input": "how can Zerodha help with investing?"})
print(result["output"])

result = agent_executor.invoke({"input": "What is the weather in Darjeeling?"})
print(result["output"])

# Step 1: Define your response function
# This function will receive the user message and return a response
def predict(message, _):
    # TODO: Add your chatbot logic here
    # For example: connect to agent_executor or call an LLM
    result = agent_executor.invoke({"input": message})
    return result["output"]


#Build the Gradio Chat Interface
# You can customize title, examples, theme, etc.
demo = gr.ChatInterface(
    fn=predict,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(
        placeholder="Hi, I am your Customer Support Bot. How can I help you today?",
        container=False,
        scale=7,
    ),
    title="Customer Support Bot",
    examples=[
        "What is the weather like in Darjeeling?",
        "How can Zerodha help in mutual fund investment?",
    ],
)

demo.launch(share=True, theme=gr.themes.Soft())



#loader = PyPDFLoader("path/to/your/file.pdf")
#documents = loader.load()

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader

loader = DirectoryLoader(
    path="path/to/your/folder",
    loader_cls=UnstructuredFileLoader
)
documents = loader.load()

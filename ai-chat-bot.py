import os
import getpass
from langchain_openai import ChatOpenAI
# Import ChatPromptTemplate to structure multi-turn conversational prompts
from langchain_core.prompts import ChatPromptTemplate
# Import a parser that extracts plain string content from LLM responses
from langchain_core.output_parsers import StrOutputParser


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OPENAI_API_KEY"] = getpass.getpass()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])


prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

while True:
    query = input("Ask something: ")
    if query.lower() in ["exit", "quit"]:
        break

    chain_result  = chain.invoke({"question": query})
    print(chain_result)

    print("\n", chain_result)
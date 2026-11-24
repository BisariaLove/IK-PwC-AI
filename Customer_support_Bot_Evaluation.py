# Extracted Python code cells from Customer_support_Bot_Evaluation.ipynb, in notebook order.
# IPython-only commands (such as !pip install) are retained as comments.

# Installing pinned stable versions
# pip install -U "requests==2.32.4"

# pip install langsmith

# Installing the LangChain OpenAI integration to use OpenAI models within LangChain workflows.
# pip install langchain-openai langchain-huggingface
# pip install sentence-transformers
# pip install langchain-classic
# Installing the core LangChain library for building LLM-based applications, including chaining, memory, and retrieval capabilities.
# pip install langchain

# Installing BeautifulSoup4 for web scraping and HTML/XML parsing, useful for extracting text from web pages.
# pip install beautifulsoup4

# Installing the community version of LangChain, which includes integrations and tools contributed by the community.
#  pip install langchain-community
# Installing FAISS (Facebook AI Similarity Search) for efficient similarity-based search on text embeddings.
# pip install faiss-cpu
#installing Tavily, a web search API used for real-time and grounded responses in LLM workflows
# pip install -U langchain-community tavily-python

# Installing Gradio, a framework to create web-based UIs for AI models and applications easily.
#  pip install -U gradio gradio_client
# pip install -U langsmith

#Sample Langsmith run - https://apac.smith.langchain.com/o/fec4ecf1-ac78-4588-a21c-b1186edcf52e/datasets/224f7c7c-2f4e-4151-af3c-d84760536fb6/compare?selectedSessions=74356a32-a612-45d3-bf99-50f2394a0484

"""
Flow of the code:
------------------

Zerodha homepage → text chunks → Hugging Face embeddings → FAISS index
                                                        ↓
Question → retrieve relevant chunks → OpenRouter model → answer + source documents

"""


import os
import bs4
from typing_extensions import Annotated, TypedDict
from langsmith import Client, traceable

from langchain_openai import ChatOpenAI
#from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from deepagents import create_deep_agent


# Supply API keys through the environment; the model calls use OpenRouter.
missing_credentials = [
    name for name in ("OPENROUTER_API_KEY", "LANGSMITH_API_KEY")
    if not os.environ.get(name)
]
if missing_credentials:
    raise RuntimeError(
        "Set required environment variables before running: "
        + ", ".join(missing_credentials)
    )


os.environ["LANGSMITH_TRACING"] = "true"

print("\n##API Keys loaded.\n")

print("\n##Loading knowledge base from Zerodha...")
loader = WebBaseLoader("https://zerodha.com/")
docs = loader.load()

print(f"Loaded {len(docs)} documents.")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)

print(f"Created {len(splits)} chunks.")

embeddings = HuggingFaceEmbeddings() #OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
#temperature controls how varied the model’s responses are.
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model= "gpt-4o-mini",#"nvidia/nemotron-3-super-120b-a12b:free", #"nvidia/nemotron-3.5-lightning:free",#"arcee-ai/trinity-mini:free",
    temperature=0.4
)


# Add decorator so this function is traced in LangSmith
@traceable()
def rag_bot(question: str) -> dict:
    # Provide a specific context instruction based on Zerodha
    docs = retriever.invoke(question) # vector DB
    docs_string = "\n\n".join(doc.page_content for doc in docs)

    instructions = f"""You are a helpful customer support assistant for Zerodha.
    Use the following retrieved context to answer the user's question.
    If you don't know the answer, just say that you don't know.
    Context: {docs_string}"""

    ai_msg = llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ])

    return {"answer": ai_msg.content, "documents": docs}

rag_bot("What is Coin?")

client = Client()

# Define the examples for the dataset
examples = [
    (
        "What exactly does Zerodha do?",
        "Zerodha is a stockbroker in India that offers a discount pricing model. They provide platforms for investing in stocks, derivatives, mutual funds, and more."
    ),
    (
        "Tell me about Coin and what it is used for.",
        "Coin is a platform by Zerodha that allows you to buy mutual funds directly from asset management companies without paying any commission."
    ),
    (
        "Is it free to start investing in mutual funds with you?",
        "Yes, direct mutual fund investments through Zerodha are absolutely free with zero commissions."
    )
]

# Create the dataset and examples in LangSmith
dataset_name = "zerodha_support_bot_eval_v3"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": q} for q, _ in examples],
        outputs=[{"answer": a} for _, a in examples],
        dataset_id=dataset.id,
    )
    print(f"Created dataset {dataset_name} with {len(examples)} examples.")
else:
    print(f"Dataset {dataset_name} already exists.")

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.
Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Explain your reasoning in a step-by-step manner to ensure your conclusion is correct."""

#grader_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(CorrectnessGrade)
grader_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model= "nvidia/nemotron-3.5-lightning",
    temperature=0.4
).with_structured_output(CorrectnessGrade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    answers = f"""QUESTION: {inputs['question']}\nGROUND TRUTH: {reference_outputs['answer']}\nSTUDENT ANSWER: {outputs['answer']}"""
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers},
    ])
    return grade["correct"]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

relevance_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION and a STUDENT ANSWER.

Do not answer outside of the context.

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Explain your reasoning step-by-step."""

#relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RelevanceGrade, method="json_schema", strict=True)
relevance_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model= "nvidia/nemotron-3.5-lightning",
    temperature=0.4
).with_structured_output(RelevanceGrade, method="json_schema", strict=True)

def relevance(inputs: dict, outputs: dict) -> bool:
    answer = f"""QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"""
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return grade["relevant"]#, grade["explanation"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning"]
    grounded: Annotated[bool, ..., "Score on if the answer hallucinates from the documents"]

grounded_instructions = """You are a teacher grading a quiz.
You will be given FACTS and a STUDENT ANSWER.
Criteria:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain hallucinatory information outside the scope of the FACTS."""

#grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
#    GroundedGrade, method="json_schema", strict=True
#)
grounded_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model= "nvidia/nemotron-3.5-lightning",
    temperature=0.4
).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

def groundedness(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"""FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"""
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": answer},
    ])
    return grade["grounded"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning"]
    relevant: Annotated[bool, ..., "True if documents are relevant to the question"]

retrieval_relevance_instructions = """You are a teacher.
You will be given a QUESTION and a set of FACTS.
Criteria:
(1) Identify FACTS that are completely unrelated.
(2) If facts contain ANY keywords or semantic meaning related to the question, consider them relevant.",
(3) It is OK if facts have SOME unrelated information as long as (2) is met."""

#retrieval_relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
#    RetrievalRelevanceGrade, method="json_schema", strict=True
#)

retrieval_relevance_llm =ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model= "nvidia/nemotron-3.5-lightning",
    temperature=0.4
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"""FACTS: {doc_string}\nQUESTION: {inputs['question']}"""
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return grade["relevant"]

def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="zerodha-eval-v4",
    metadata={"version": "GPT-4o-mini web-scraped base"},
)

experiment_results.to_pandas()

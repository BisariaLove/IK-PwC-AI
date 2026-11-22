import getpass
import os
import warnings

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPEN_API_KEY"] = getpass.getpass()


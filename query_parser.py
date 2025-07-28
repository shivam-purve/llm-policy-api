from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt_template = ChatPromptTemplate.from_template("""
Extract the following fields from the query:
- age
- gender
- procedure
- location
- policy_duration

Query: {query}

Respond in JSON only.
""")

def parse_query(query):
    prompt = prompt_template.format_messages(query=query)
    response = llm(prompt)
    return response.content

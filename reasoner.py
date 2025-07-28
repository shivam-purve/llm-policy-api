from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0, model="gpt-4")

reasoning_prompt = ChatPromptTemplate.from_template("""
Based on the following insurance clauses:

{clauses}

And this query (structured):
{query}

Determine if the insurance claim should be approved, what payout amount applies, and why.

Respond in JSON format:
{{
  "decision": "...",
  "amount": "...",
  "justification": "..."
}}
""")

def generate_decision(structured_query, relevant_docs):
    clause_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = reasoning_prompt.format_messages(query=structured_query, clauses=clause_text)
    response = llm(prompt)
    return response.content

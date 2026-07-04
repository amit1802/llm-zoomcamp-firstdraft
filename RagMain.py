from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import ingest
from minsearch import Index
from rag_helper import RAGBase

load_dotenv()

open_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
    )

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""
USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""
documents = ingest.load_faq_data()
index=ingest.build_index(documents)

rag_answer =  RAGBase(
    index = index,
    llm_client = open_client,
    #instructions = INSTRUCTIONS,
    #prompt_template = USER_PROMPT_TEMPLATE
    #course = 'llm-zoomcamp',
    #model = 'meta-llama/llama-3.1-8b-instruct'
)

query = 'I just discovered the course. Can I still join and can I use any AI model? Or do we have to work with just openAI'
answer = rag_answer.rag(query)
print(answer)

# def search(question,course="llm-zoomcamp"):
#     boost_dict = {"question":2.0,"section":0.5}
#     filter_dict = {"course":course}

#     return index.search(
#         question,
#         boost_dict = boost_dict,
#         filter_dict = filter_dict,
#         num_results=5
#     )

# def build_context(search_results):
#     lines = []
#     for doc in search_results:
#         lines.append(doc["section"])
#         lines.append("Q. "+doc["question"])
#         lines.append("A. "+doc["answer"])
#         lines.append("")
    
#     return "\n".join(lines).strip()

# def build_prompt(question, search_results):
#     context = build_context(search_results)
#     prompt = USER_PROMPT_TEMPLATE.format(
#         question = question,
#         context = context
#     )
#     return prompt.strip()

# def llm(instructions,user_prompt, model="google/gemma-4-31b-it:free"):
#         message_history = [{"role":"developer", "content":instructions}, {"role":"user","content":user_prompt}]
        
#         response = open_client.chat.completions.create(
#         #response = open_client.responses.create(
#                 #model="google/gemini-2.5-flash",
#                 model=model,
#                 messages=message_history
#                 #input=message_history
#                 #max_output_tokens=128    
#         )
#         return response.choices[0].message.content
#         #return response.output_text

# def rag(query, model="meta-llama/llama-3.1-8b-instruct"):
#         search_results=search(query)
#         prompt = build_prompt(query, search_results)
#         print(prompt)
#         answer = llm(INSTRUCTIONS,prompt,model=model)
#         return answer

# #answer = rag("I just discovered the course. Can I join now?")
# answer = rag("How do I run Ollama?")
# #rag("OpenSource: Can I use open-source alternatives to OpenAI API?")
# #answer = rag("If my final project [or the basic idea behind the project] is same as any project already submitted in this cohort or previous cohorts, then will my project still be accepted? If not is there any way we can declare our project ideas, before we start working on the project?")
# print(answer)

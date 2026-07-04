INSTRUCTIONS = '''Your task is to find the answers from the course participants based on the provided contex.
Use the context to find relevant information and provide accurate answers.
If te answer is not found in the context reply with "I don't know :)". 
'''
PROMPT_TEMPLATE = '''
Question: {question}
Context: {context}
'''
class RAGBase:

    def __init__(
            self,
            index,
            llm_client,
            instructions = INSTRUCTIONS,
            prompt_template = PROMPT_TEMPLATE,
            course = "llm-zoomcamp",
            #model = "google/gemma-4-31b-it:free"
            model = "google/gemini-2.5-flash"

    ):
            self.index = index
            self.llm_client = llm_client
            self.instructions = instructions
            self.prompt_template = prompt_template
            self.course = course
            self.model = model
        
    def search(self,question:str,course:str="llm-zoomcamp"):
        boost_dict = {"question":2.0,"section":0.5}
        filter_dict = {"course":self.course}

        return self.index.search(
            question,
            boost_dict = boost_dict,
            filter_dict = filter_dict,
            num_results=5
         )
    
    def build_context(self,results:list) -> str:
        context_lines = []
        for doc in results:
            context_lines.append(doc["section"])
            context_lines.append("Q. "+ doc["question"])
            context_lines.append("A. "+ doc["answer"])
            context_lines.append('')
        
        return '\n'.join(context_lines).strip()

    def build_prompt(self, question:str, search_results:list) -> str:
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(question = question, context = context)

        return prompt
    
    def llm(self, prompt):
        message_history = [
            {"role" : "developer", "content" : self.instructions},
            {"role" : "user", "content" : prompt}
        ]

        response = self.llm_client.responses.create(
            model = self.model,
            input = message_history,
            max_output_tokens=256
        )

        return response.output_text
    
    def rag(self, question):
        search_results = self.search(question)
        prompt = self.build_prompt(question, search_results)
        #print(prompt)
        #print("Now Answer - ")
        return self.llm(prompt)
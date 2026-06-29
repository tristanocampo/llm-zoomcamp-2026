# SEARCH
# BUILD PROMPT
# CALL LLM


from google.genai import types


INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context and frequently asked questions, please
respond with "I don't know."
'''


USER_PROMPT_TEMPALATE = '''
Question:
{question}

Context:
{context}
'''


class RAGBase:
    def __init__(
            self,
            index, 
            llm_client,
            llm_model="gemini-3.1-flash-lite",
            instructions=INSTRUCTIONS,
            prompt_template = USER_PROMPT_TEMPALATE,
            course = 'llm-zoomcamp'

    ):
        self.index = index
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course


    def search(self, query: str, course: str = 'llm-zoomcamp') -> list:
        """
        Search the FAQ database for entries matching the given query.

        Args:
            query: Search query text to look up in the course FAQ.
            course: The course for which to search FAQs.
        """
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": course}

        return self.index.search(
            query,
            num_results=5,
            boost_dict=boost_dict,
            filter_dict=filter_dict
    )


    search_function = types.FunctionDeclaration(
        name="search",
        description="Search the Frequently Asked Questions database for entries matching the query.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    )

    search_tool = types.Tool(function_declarations=[search_function])


    def build_context(self, search_results):
        '''Builds a context string from the search results, and the prompt template
           to be used in the prompt for the LLM.'''     

        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()



    def build_prompt(self, question, search_results):
        '''Builds a prompt for the LLM using the provided question and search results.'''

        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )   
        return prompt.strip()
    


    def llm_gemini(self, prompt):
        '''Calls the LLM with the provided prompt and returns the response.
        - Uses the instructions and the prompt to create a message history for the LLM.
        - Uses Gemini API Client to call the LLM and get the response.'''

        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config = types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools = [self.search_tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
            )
        )

        return response.text.strip()
    
    def ask(self, question):
        '''Performs the RAG process for the provided question.'''

        search_results = self.search(question)
        prompt = self.build_prompt(question, search_results)
        answer = self.llm_gemini(prompt)

        return answer.strip()
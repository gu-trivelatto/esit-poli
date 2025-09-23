from abc import ABC, abstractmethod

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from src.libs.state import GraphStateType
from src.libs.memory import Memory


class ResearchAgentBase(ABC):
    def __init__(self, llm_models, retriever, web_tool, state: GraphStateType, app, debug):
        self.retriever = retriever
        self.web_tool = web_tool
        self.chat_model = llm_models['chat_model']
        self.json_model = llm_models['json_model']
        self.ht_model = llm_models['ht_model']
        self.ht_json_model = llm_models['ht_json_model']
        self.state = state
        self.debug = debug
        self.app = app
        self.selected_value = None
        self.memory = Memory()
        
    def get_answer_analyzer_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at summarizing a bunch of data to extract only the important bits from it.

            Given the user's QUERY and the SEARCH_RESULTS, summarize as briefly as possible the information
            searched by the user. Don't give any preamble or introduction, go directly to the summary
            of the requested information.
            
            If it helps to provide a more precise answer, you can also make use of the CONTEXT.
            
            Whenever there is a source in the data received in SEARCH_RESULTS, you MUST include the used
            sources as a topic at the end of the summarization. In case there is no source, simply ignore this
            instruction.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            SEARCH_RESULTS: {search_results} \n
            CONTEXT: {context} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","search_results","context"],
        )

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphStateType:
        return self.state
    
class ResearchInfoWeb(ResearchAgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a master at working out the best keywords to search for in a web search to get the best info for the user.

            Given the QUERY, work out the best search queries that will find the info requested by the user
            The queries must contain no more than a phrase, since it will be used for online search. 

            Return a JSON with a single key 'keywords' with at most 3 different search queries.
            
            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Searching info in the internet')
        if self.debug:
            self.memory.save_debug("---RESEARCH INFO SEARCHING---")
            
        query = self.state['next_query']['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        prompt = self.get_prompt_template()
        answer_analyzer_prompt = self.get_answer_analyzer_prompt_template()
        
        llm_chain = prompt | self.json_model | JsonOutputParser()
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()

        # Web search
        keywords = llm_chain.invoke({"query": query, "context": context})
        keywords = keywords['keywords']
        full_searches = []
        for idx, keyword in enumerate(keywords):
            temp_docs = self.web_tool.execute(keyword)
            web_results = ''
            if type(temp_docs) == list:
                for d in temp_docs:
                    web_results += f'Source: {d["url"]}\n{d["content"]}\n'
                web_results = Document(page_content=web_results)
            elif type(temp_docs) == dict:
                web_results = f'\nSource: {temp_docs["url"]}\n{temp_docs["content"]}'
                web_results = Document(page_content=web_results)
            else:
                web_results = 'No results'
            if self.debug:
                self.memory.save_debug(f'KEYWORD {idx}: {keyword}')
                self.memory.save_debug(f'RESULTS FOR KEYWORD {idx}: {web_results}')
            if full_searches is not None:
                full_searches.append(web_results)
            else:
                full_searches = [web_results]

        processed_searches = answer_analyzer_chain.invoke({"query": query, "search_results": full_searches, "context": context})
        
        if self.debug:
            self.memory.save_debug(f'FULL RESULTS: {full_searches}\n')
            self.memory.save_debug(f'PROCESSED RESULT: {processed_searches}\n')
        
        self.state['context'] = context + [processed_searches]
        self.state['num_steps'] = num_steps
        
        return self.state
    
# TODO check the answer analyzer prompt
# TODO create chain to decide whether to search information on the paper or on the CESM documentation

class ResearchInfoRAG(ResearchAgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a master at working out the best questions to ask our knowledge agent to get 
            the best info for the customer. \n

            Given the QUERY, work out the best questions that will find the best info for 
            helping to write the final answer. Write the questions to our knowledge system not 
            to the customer. \n

            Return a JSON with a single key 'questions' with no more than 3 strings of and no 
            preamble or explaination. \n
            
            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Consulting the source paper')
        question_rag_prompt = self.get_prompt_template()
        answer_analyzer_prompt = self.get_answer_analyzer_prompt_template()
        
        question_rag_chain = question_rag_prompt | self.json_model | JsonOutputParser()
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()
        
        if self.debug:
            self.memory.save_debug("---RAG PDF PAPER RETRIEVER---")
            
        query = self.state['consolidated_input']
        context = self.state['context']
        action_history = self.state['action_history']
        num_steps = self.state['num_steps']
        num_steps += 1

        questions = question_rag_chain.invoke({"query": query})
        questions = questions['questions']

        rag_results = []
        for idx, question in enumerate(questions):
            temp_docs = self.retriever.execute(question)
            if self.debug:
                self.memory.save_debug(f'QUESTION {idx}: {question}')
                self.memory.save_debug(f'ANSWER FOR QUESTION {idx}: {temp_docs.response}')
            question_results = question + '\n\n' + temp_docs.response + "\n\n\n"
            if rag_results is not None:
                rag_results.append(question_results)
            else:
                rag_results = [question_results]
        if self.debug:
            self.memory.save_debug(f'FULL ANSWERS: {rag_results}\n')
        
        processed_searches = answer_analyzer_chain.invoke({"query": query, "search_results": rag_results, "context": context})
        # TODO find a way of referencing the used pdfs
        result = f'Source: PDF paper \n{query}: \n{processed_searches}'
        
        action_history['consult'] = 'done'
        self.state['action_history'] = action_history
        self.state['context'] = context + [result]
        self.state['num_steps'] = num_steps
        
        return self.state

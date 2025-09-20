from datetime import datetime
from abc import ABC, abstractmethod
from src.libs.state import GraphStateType
from src.libs.helper import HelperFunctions
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

# TODO standardize variable names used from the state
# TODO standardize the way the agents interact with the state

class AgentBase(ABC):
    def __init__(self, llm_models, es_models, state: GraphStateType, app, debug):
        self.chat_model = llm_models['chat_model']
        self.json_model = llm_models['json_model']
        self.ht_model = llm_models['ht_model']
        self.ht_json_model = llm_models['ht_json_model']
        self.state = state
        self.debug = debug
        self.app = app
        self.selected_value = None
        self.base_model = es_models['base_model']
        self.mod_model = es_models['mod_model']
        self.helper = HelperFunctions()
        
    def confirm_selection(self, selected_value):
        self.selected_value = selected_value

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphStateType:
        return self.state
    
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
        self.helper = HelperFunctions()
        
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

class DateGetter(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphStateType:
        self.helper.save_chat_status('Processing')
        num_steps = self.state['num_steps']
        num_steps += 1
        
        current_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        result = f'The current date and time are {current_date}'
        
        if self.debug:
            self.helper.save_debug("---DATE GETTER TOOL---")
            self.helper.save_debug(f'CURRENT DATE: {current_date}\n')

        self.state['context'] = [result]
        self.state['num_steps'] = num_steps

        return self.state

class InputTranslator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are responsible of verifying if the USER_INPUT is in another language other
            than english, if so, translate the input. If the text is already in english
            simply output the same text. \n
            
            Your output must be a JSON object with two keys, 'language' and 'input', where
            'language' is the source language the user wrote and 'input' is the translated
            input. \n
        
            If the language is english, use ALWAYS the whole word to select it, never 'en',
            'eng', or any other variation. Use ALWAYS 'english'. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input"],
        )
    
    def execute(self) -> GraphStateType:
        user_input = self.state['user_input']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_json_model | JsonOutputParser()

        llm_output = llm_chain.invoke({"user_input": user_input})
        translated_user_input = llm_output['input']
        source_language = llm_output['language']
        
        if self.debug:
            self.helper.save_debug("---TRANSLATE INPUT---")
            self.helper.save_debug(f'ORIGINAL INPUT: {user_input.rstrip()}')
            self.helper.save_debug(f'SOURCE LANGUAGE: {source_language}')
            if source_language.lower() != 'english':
                self.helper.save_debug(f'TRANSLATED INPUT:{translated_user_input.rstrip()}\n')
        
        self.state['num_steps'] = num_steps
        self.state['user_input'] = translated_user_input
        self.state['target_language'] = source_language
        
        return self.state

class ToolBypasser(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), you are responsible for checking
            if the user trying to have a simple interaction with the tool instead of search for
            information or manipulate the model. \n

            Considering the USER_INPUT you should decide whether to route
            the user to the tool or simply bypass it to generate a simple answer to the user. \n
            
            The cases where you will bypass to the output are when USER_INPUT contains:
            - 'Hello!', 'Hi!', 'Hey!' or similars, without anything else;
            - 'How are you?' or similars, without anything else;
            - 'Who are you?' or similars;
            - 'What do you do?' or similars;
            - 'Thank you!', 'Thanks!', 'Thanks a lot!' or similars;
            
            You must output a JSON with a single key 'is_conversation' containing exclusivelly the 
            selected type, which can be either true or false (use full lowercase for the boolean). \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input"],
        )
    
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"user_input": user_input})
        is_conversation = llm_output['is_conversation']
        if type(is_conversation) == str:
            is_conversation = True if is_conversation == 'true' else False
        elif type(is_conversation) != bool:
            is_conversation = False
        
        if self.debug:
            self.helper.save_debug("---TYPE IDENTIFIER---")
            self.helper.save_debug(f'BYPASS TO OUTPUT: {is_conversation}\n')
        
        self.state['is_conversation'] = is_conversation
        self.state['num_steps'] = num_steps
        
        return self.state

class TypeIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), your main task is to
            initiate the pipeline of the tool by deciding which type of request the user
            is trying to ask. You have three execution branches available, "energy_system",
            "mixed" and "general". \n

            "energy_system": the USER_INPUT is related to modeling. Unless the user specifies that the
            modeling he is talking about is not related to the model this tool is designed to
            analyze, you will use this branch. This branch allows you to complete diverse tasks
            related to Energy System modeling, like consulting information about the model,
            modifying values of the model, running it, comparing and plotting results, and also
            consults to the paper related to the model for more details. Anything on these lines
            should use this branch. You should also use this branch when the user talks about
            technical parameters, parametrization in general and asks about cientific data
            related to the development of the paper/model. Inputs without previous context pointing
            to the model that talk about changes, modifications, and other stuff related to 
            comparisons in different years normally are also related to the energy system. \n
            
            "mixed": the USER_INPUT would be identified as "energy_system", but there are references in
            the input that must be researched online to be able to gather the necessary context for the
            modelling tools to reach the user's goals. \n
            
            "general": the USER_INPUT is related to some generic topic, it may consist of one or more
            points that require searching for information. \n
            
            You must output a JSON with a single key 'input_type' containing exclusivelly the 
            selected type. \n
            
            You may also use the CHAT_HISTORY to get a better context of past messages exchanged
            between you and the user, and better understand what he wants in the current message. \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","history"],
        )
    
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"user_input": user_input, "history": history})
        selected_type = llm_output['input_type']
        
        if self.debug:
            self.helper.save_debug("---TYPE IDENTIFIER---")
            self.helper.save_debug(f'USER INPUT: {user_input.rstrip()}')
            self.helper.save_debug(f'IDENTIFIED TYPE: {selected_type}\n')
            self.helper.save_debug(history)
        
        self.state['input_type'] = selected_type
        self.state['num_steps'] = num_steps
        
        return self.state
    
class InputConsolidator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at verifying if the USER_INPUT is complete on it's own or if it
            references data from the CHAT_HISTORY. \n
            
            You should output a JSON object with a single key 'consolidated_input', in which
            you have two possible situations.
            1. The USER_INPUT don't reference past messages and the following tools can use
            it as it is to execute their actions;
            2. The USER_INPUT references messages from CHAT_HISTORY, and this information is
            needed for the following tools. \n
            
            Your actions based on the situations are:
            1. Simply repeat the USER_INPUT as the 'consolidated_input';
            2. Build a new 'consolidated_input' adding the necessary information from the
            CHAT_HISTORY, while still keeping it as similar as possible with USER_INPUT. \n
            
            When modifying the input you must restrain yourself to substituting references to
            specific data by the data itself, for example, if the user talked about a specific
            parameter in the past message, and now said something like 'now modify it to X', you
            should substitute 'it' by the parameter name. Never add context related to results
            or general information. You should keep your rewriting as minimal as possible. \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input", "history"],
        )
        
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "history": history})
        consolidated_input = llm_output['consolidated_input']
        
        if self.debug:
            self.helper.save_debug("---INPUT CONSOLIDATOR---")
            self.helper.save_debug(f'USER INPUT: {user_input.rstrip()}')
            self.helper.save_debug(f'CONSOLIDATED INPUT: {consolidated_input}\n')
            
        self.state['consolidated_input'] = consolidated_input
        self.state['num_steps'] = num_steps
        
        return self.state

class QueryGenerator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at developing a query to retrieve information from other means
            given the USER_INPUT and the CONTEXT with already known information. \n
            
            Using the CONTEXT to check which information is already available, you should analyze the
            USER_INPUT and decide what will be the next query for the available tools. You have 
            three options to gather information, online search, calculator and the aditional
            user's inputs. The identifiers of these options are ['web_search', 'calculator', 'user_input']. \n
            
            You'll also receive QUERY_HISTORY, use it to avoid repeating questions. If similar information was
            already searched at least 3 times without success, you may request for the user to input more
            information about what he wants. You may also check for the CHAT_HISTORY to verify what already
            happened in this session of the chat with the user, some information may be already present there. \n
            
            Restrain yourself as much as possible to request the user for more data, the only two cases you
            may do this are:
            1. The user asked for you to use information that only he knows (personal information for example);
            2. You already tried searching the information at least 3 times but couldn't find it, then you may
            prompt the user if the information could be provide more context for further research. If you couldn't
            find even having the necessary context, simply select 'no_action' as the selected tool. \n
            
            Also, never consider that you know how to to calculations, if there is any calculation in the 
            user request, build a query with it and pass it forward. For any mathematical problem you should use
            'calculator'. Be sure that you have all the necessary data before routing to this tool. \n
            
            You must output a JSON object with two keys, 'tool' and 'next_query'. 'tool' may contain one of the
            following ['web_search', 'calculator', 'user_input', 'no_action'] while 'next_query' is the next
            query to be processed by the tools. \n
            
            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            QUERY_HISTORY: {query_history} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","query_history","history"],
        )
        
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        ## Get the state
        user_input = self.state['user_input']
        context = self.state['context']
        query_history = self.state['query_history']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input,
                                   "context": context,
                                   "query_history": query_history,
                                   "history": history
                                   })
        
        if self.debug:
            self.helper.save_debug("---CONTEXT ANALYZER---")
            self.helper.save_debug(f'NEXT QUERY: {llm_output}\n')
            self.helper.save_debug(history)
        
        self.state['query_history'] = query_history + [llm_output['next_query']]
        self.state['next_query'] = llm_output
        self.state['num_steps'] = num_steps
        
        return self.state

class Mixed(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading the user USER_INPUT and the available CONTEXT to decide if there
            is already enough information gathered to fulfill the energy system related command
            made by the user. \n
            
            You must be certain that you have all the data before deciding to send it to the
            modelling section of the pipeline. If any of the values asked by the user is not
            directly given by him, you can't consider the data complete unless you have the
            desired value in the CONTEXT. \n
            
            You may also check for the CHAT_HISTORY to verify what already happened in this session
            of the chat with the user. \n

            You must output a JSON object with a single key 'is_data_complete' containing a boolean
            on whether you have enough data for the user's request or not. \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            CONTEXT: {context} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","history"],
        )
        
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "history": history})
        is_data_complete = llm_output['is_data_complete']
        
        if self.debug:
            self.helper.save_debug("---TOOL SELECTION---")
            self.helper.save_debug(f'USER INPUT: {user_input.rstrip()}')
            self.helper.save_debug(f'CONTEXT: {context}')
            self.helper.save_debug(f'DATA IS COMPLETE: {is_data_complete}\n')
            
        self.state['is_data_complete'] = is_data_complete
        self.state['num_steps'] = num_steps
        
        return self.state

class ContextAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at analyzing the available CONTEXT and CHAT_HISTORY to decide if the available
            information is already enough to answer the question asked in USER_INPUT. \n

            It's important to know that you are the context analyzer for a branch of a tool, and this branch is
            responsible to gather general information to be used in the modeling context. If the USER_INPUT
            is related to modeling and uses information available at the data sources to apply modeling actions
            and all this information is available, then you should consider that the context is ready. Other
            tools will be responsible of using this data for modeling. \n
            
            If there is nothing related to modeling you can simply define it as ready when all information is
            gathered. \n
            
            Your output is a JSON object with a single key 'ready_to_answer', where you can use either true
            or false (always write it in lowercase). \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT : {context} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","history"]
        )
        
    def execute(self) -> GraphStateType:
        self.helper.save_chat_status('Processing')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['consolidated_input']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "history": history})
        
        if self.debug:
            self.helper.save_debug("---TOOL SELECTION---")
            self.helper.save_debug(f'READY TO ANSWER: {llm_output["ready_to_answer"]}\n')
        
        self.state['next_query'] = llm_output
        self.state['num_steps'] = num_steps
        
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
        self.helper.save_chat_status('Searching info in the internet')
        if self.debug:
            self.helper.save_debug("---RESEARCH INFO SEARCHING---")
            
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
                self.helper.save_debug(f'KEYWORD {idx}: {keyword}')
                self.helper.save_debug(f'RESULTS FOR KEYWORD {idx}: {web_results}')
            if full_searches is not None:
                full_searches.append(web_results)
            else:
                full_searches = [web_results]

        processed_searches = answer_analyzer_chain.invoke({"query": query, "search_results": full_searches, "context": context})
        
        if self.debug:
            self.helper.save_debug(f'FULL RESULTS: {full_searches}\n')
            self.helper.save_debug(f'PROCESSED RESULT: {processed_searches}\n')
        
        self.state['context'] = context + [processed_searches]
        self.state['num_steps'] = num_steps
        
        return self.state

class Calculator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an specialist at identifying the correct operation and operands to build
            a JSON object with the correct equation to be solved, you will get the information of
            what equation to generate from the given QUERY. \n
            
            You must output a JSON object with a single key 'equation', that must be in a format
            solvable with Python, since we'll solve this using the eval() function. You don't need
            to import anything or prepare the environment in any way, the environment is ready,
            you just need to build the equation in a format recognizable by python, using python
            mathematical functions if necessary. \n
            
            NEVER USE IMPORT STATEMENTS, YOU MUST ASSUME THAT ALL NECESSARY LIBRARIES ARE ALREADY
            IMPORTED, JUST USE THEM. \n
            
            If you need information that you don't know to build the equation you may find that
            in the CONTEXT, so check it for more details on the data. \n
            
            If you need intermediate results, you can calculate only them and leave the final
            result for later, you're part of a iterative tool that may be called as many times
            as needed to find the final result. \n
            
            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            CONTEXT: {context}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","context"],
        )
        
    def execute(self) -> GraphStateType:
        self.helper.save_chat_status('Calculating result')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
    
        query = self.state['next_query']['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"query": query, "context": context})
        equation = llm_output['equation']
        
        if self.debug:
            self.helper.save_debug("---CALCULATOR TOOL---")
            self.helper.save_debug(f'EQUATION: {equation}')
        
        result = eval(equation)
        
        str_result = f'{equation} = {result}'
        
        if self.debug:
            self.helper.save_debug(f'RESULT: {str_result}\n')
            
        self.state['context'] = context + [str_result]
        self.state['num_steps'] = num_steps
        
        return self.state

class InfoTypeIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of a modelling insights system, in this system we have two sources of
            information, the paper that explains the model that we use and the model data itself.
            You are an expert at defining from which of these two the information asked by the
            user should be gathered. \n
            
            You have access to the USER_INPUT, and based on that you should decide if the output
            is 'model' or 'paper'. \n
            
            Normally, the output will be 'model' if the user asks for the value of a specific
            parameter, or if the user asks about components that are modeled (such as commodity,
            conversion processes, conversion subprocesses and scenario information) in a general
            way (not asking for detailed modelling information). \n
            
            On other cases you should use 'paper', that's where the theoric information behind
            the model lies, information such as intrinsec details of the modeling process,
            constraints, parameter relationships, details of the types of parameters and 
            implementation details. \n

            Return a JSON with a single key 'type' that contains either 'model' or 'paper'. \n
            
            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input"],
        )
    
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['consolidated_input']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input})
        
        if self.debug:
            self.helper.save_debug("---INFO TYPE IDENTIFIER---")
            self.helper.save_debug(f'RETRIEVAL TYPE: {llm_output["type"]}\n')
        
        self.state['retrieval_type'] = llm_output['type']
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
        self.helper.save_chat_status('Consulting the source paper')
        question_rag_prompt = self.get_prompt_template()
        answer_analyzer_prompt = self.get_answer_analyzer_prompt_template()
        
        question_rag_chain = question_rag_prompt | self.json_model | JsonOutputParser()
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()
        
        if self.debug:
            self.helper.save_debug("---RAG PDF PAPER RETRIEVER---")
            
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
                self.helper.save_debug(f'QUESTION {idx}: {question}')
                self.helper.save_debug(f'ANSWER FOR QUESTION {idx}: {temp_docs.response}')
            question_results = question + '\n\n' + temp_docs.response + "\n\n\n"
            if rag_results is not None:
                rag_results.append(question_results)
            else:
                rag_results = [question_results]
        if self.debug:
            self.helper.save_debug(f'FULL ANSWERS: {rag_results}\n')
        
        processed_searches = answer_analyzer_chain.invoke({"query": query, "search_results": rag_results, "context": context})
        # TODO find a way of referencing the used pdfs
        result = f'Source: PDF paper \n{query}: \n{processed_searches}'
        
        action_history['consult'] = 'done'
        self.state['action_history'] = action_history
        self.state['context'] = context + [result]
        self.state['num_steps'] = num_steps
        
        return self.state

# TODO the outputs should also indicate if the model was runned etc...
    
class OutputGenerator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), a tool tailored
            to use LLM agents to help users to analyze, manipulate and understand
            energy system models. You are the last agent of the tool, responsible
            for summing up the results generated by other agents in the output
            to be given to the user. If the user asks about you, present yourself
            as the tool itself, not as an agent. \n
            
            Given the USER_INPUT and a CONTEXT, generate an answer for the query
            asked by the user. You should make use of the provided information
            to answer the user in the best possible way. If you think the answer
            does not answer the user completely, ask the user for the necessary
            information if possible. \n
            
            Since you are part of a model analyzing tool you will probably receive
            information about model modifications, model runs, model results, etc.
            If the user is talking about manipulations in the model you should 
            restrain your answer to what was actually done regarding the model
            (information that you'll find in CONTEXT and ACTION_HISTORY), never
            make up modifications or indicate plots for yourself. Other nodes
            already manipulated the model and showed plots to the user, you should
            just forward this information to the user. For listing information
            about parameters you should show a topic list to the user if possible.
            Your main goal in this kind of scenario is to not only tell the user
            what was done to the model, but also, provide as much information as
            possible regarding the model results. Never sum up so much that the
            details about simulation results are lost. \n
            
            CHAT_HISTORY can also be used to gather context and information about
            past messages exchanged between you and the user. \n
            
            Also, you can look at the ACTION_HISTORY to understand which actions were
            executed regarding the model. \n
            
            If and only if the context you use to answer the user provides you a
            data source you should display the provided sources as follows:
            Source:
            - <url/document>
            - <url/document>
            - And so on for how many sources are needed... \n
            
            NEVER output an empty source, if there is no provided source DON'T
            ADD THE SOURCE LIST. THE PRESENCE OF THE SOURCE LIST IS OPTIONAL,
            IF THERE ARE NO SOURCES IN THE DATA YOU USED TO GENERATE THE ANSWER, THEN
            DON'T CREATE THE SOURCE LIST. \n            
            
            It's important never to cite the variables you receive, answer the most
            naturally as possible trying to make it as it was a simple conversation. \n
            
            You will always have access to the current time, but don't mention it unless
            necessary for what the user asked to the tool. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            ACTION_HISTORY: {action_history} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","action_history","history"],
        )
        
    def execute(self) -> GraphStateType:
        self.helper.save_chat_status('Generating output')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_model | StrOutputParser()
        
        ## Get the state
        user_input = self.state['user_input']
        context = self.state['context']
        action_history = self.state['action_history']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "action_history": action_history, "history": history})
        
        if self.debug:
            self.helper.save_debug("---GENERATE OUTPUT---")
            self.helper.save_debug(f'GENERATED OUTPUT:\n{llm_output}\n')
        
        if '\nSource:\n- None' in llm_output:
            llm_output = llm_output.replace('\nSource:\n- None','')
            
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = llm_output
        
        return self.state
    
class OutputTranslator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are the final node of a tool, and you are responsible for translating the
            TOOL_OUTPUT from english to a given TARGET_LANGUAGE. \n
            
            Your output must be a JSON object with a single key, 'output', where you should
            put the translated output. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            TOOL_OUTPUT : {tool_output} \n
            TARGET_LANGUAGE : {target_language} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["tool_output", "target_language"],
        )
    
    def execute(self) -> GraphStateType:
        final_answer = self.state['final_answer']
        target_language = self.state['target_language']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        if self.debug:
            self.helper.save_debug("---TRANSLATE OUTPUT---")
            self.helper.save_debug(f'TARGET LANGUAGE: {target_language}\n')
        
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_json_model | JsonOutputParser()

        llm_output = llm_chain.invoke({"tool_output": final_answer, "target_language": target_language})
        
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = llm_output['output']
        
        return self.state
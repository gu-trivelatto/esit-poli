from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from src.libs.state import GraphStateType
from src.libs.agents.main_agents import AgentBase


class InputTranslator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are responsible of verifying if the USER_INPUT is in any language other
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
            self.memory.save_debug("---TRANSLATE INPUT---")
            self.memory.save_debug(f'ORIGINAL INPUT: {user_input.rstrip()}')
            self.memory.save_debug(f'SOURCE LANGUAGE: {source_language}')
            if source_language.lower() != 'english':
                self.memory.save_debug(f'TRANSLATED INPUT:{translated_user_input.rstrip()}\n')
        
        self.state['num_steps'] = num_steps
        self.state['user_input'] = translated_user_input
        self.state['target_language'] = source_language
        
        return self.state
    
class ToolSelector(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), the execution until now decided
            that the user input requires the use of a tool to be answered, however, before using
            any tool, you need to decide which tool is the most appropriate for the user request. \n
            
            The following tools are available for you to use: \n
            1. web_search: Use this tool when the user is looking for general information that
               can be found on the internet, such as news, articles, or general knowledge.
            2. calculator: Use this tool when the user is asking for mathematical calculations,
               such as arithmetic operations, algebra, or any other type of mathematical problem.
            3. rag_search: Use this tool when the user is looking for specific information
               that can be found in the provided documents, such as reports, data sheets, or
               any other type of document provided as a data source.
            4. consult_data: Use this tool when the user is looking for specific data
               related to the energy system, such as consumption data, production data, or
               any other type of data that can be found in the system's database. \n
            5. none: Use this option when none of the tools are required to answer the user input,
               and you can answer the question directly. \n
            
            Your output must be a JSON object with a single key 'selected_tool', where the value
            must be the name of the selected tool as it appears in the provided list. \n
            
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
        selected_tool = llm_output['selected_tool']
        
        if self.debug:
            self.memory.save_debug("---TOOL SELECTOR---")
            self.memory.save_debug(f'SELECTED TOOL: {selected_tool}\n')
        
        self.state['selected_tool'] = selected_tool
        self.state['num_steps'] = num_steps
        
        return self.state

class ContextAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at analyzing the available CONTEXT and CHAT_HISTORY to decide if the available
            information is already enough to answer the question asked in USER_INPUT. \n
            
            If the context informs you that an action could not be performed, or if the
            context starts repeating itself, indicating that it got stuck in a loop, you must
            decide that the answer is ready to avoid infinite loops. \n
            
            For data gathering and plotting tasks, the text [PLOT SHOWN] in the context
            indicates that the plot was successfully shown to the user. If it was the
            only action requested, you can assume it is ready. \n
            
            Your output must be only 'ready' or 'continue', all lower case without
            any backticks, where ready means that all actions that needed to be
            taken already were, and continue means that there are still actions
            that need to be taken.\n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT : {context} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","history"]
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Processing')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.chat_model | StrOutputParser()
        
        user_input = self.state['user_input']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "history": history})
        
        if self.debug:
            self.memory.save_debug("---CONTEXT ANALYZER---")
            self.memory.save_debug(f'READY TO ANSWER: {llm_output}\n')
        
        self.state['is_data_complete'] = llm_output.lower() == "ready"
        self.state['num_steps'] = num_steps
        
        return self.state
    
class OutputTranslator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are the final node of a tool, and you are responsible for translating the
            TOOL_OUTPUT from english to a given TARGET_LANGUAGE. \n
            
            The tool name is Energy System Insight Tool (ESIT), never translate this name. \n
            
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
            self.memory.save_debug("---TRANSLATE OUTPUT---")
            self.memory.save_debug(f'TARGET LANGUAGE: {target_language}\n')
        
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_json_model | JsonOutputParser()

        llm_output = llm_chain.invoke({"tool_output": final_answer, "target_language": target_language})
        
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = llm_output['output']
        
        return self.state

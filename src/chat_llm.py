from abc import ABC
from IPython.display import Image, display

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

import src.libs.agents.main_agents as main_agents
import src.libs.agents.flow_agents as flow_agents
import src.libs.agents.tool_agents as tool_agents
import src.libs.agents.research_agents as research_agents

import src.libs.routers as routers
import src.libs.printers as printers
from src.libs.state import GraphStateType

from src.tools.web_search import WebSearchTool
from src.tools.rag_retriever import RAGRetriever

from src.config.models import Models


class GraphBuilder(ABC):
    def __init__(self, model_path, app, debug):
        self.llm_models = Models()

        self.retriever = RAGRetriever(self.llm_models.chat_model)
        self.web_tool = WebSearchTool()
        
        self.es_models = {'base_model': f'{model_path}/DEModel.xlsx',
                          'mod_model': f'{model_path}/DEModel_modified.xlsx'}
        self.debug = debug
        self.app = app
    
    # Agents (Nodes of the Graph)
    
    def input_translator(self, state: GraphStateType) -> GraphStateType:
        return flow_agents.InputTranslator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def tool_bypasser(self, state: GraphStateType) -> GraphStateType:
        return flow_agents.ToolBypasser(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def tool_selector(self, state: GraphStateType) -> GraphStateType:
        return flow_agents.ToolSelector(self.llm_models, self.es_models, state, self.app, self.debug).execute()

    def research_info_web(self, state: GraphStateType) -> GraphStateType:
        return research_agents.ResearchInfoWeb(self.llm_models, self.retriever, self.web_tool, state, self.app, self.debug).execute()

    def calculator(self, state: GraphStateType) -> GraphStateType:
        return tool_agents.Calculator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def context_analyzer(self, state: GraphStateType) -> GraphStateType:
        return flow_agents.ContextAnalyzer(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def rag_search(self, state: GraphStateType) -> GraphStateType:
        return research_agents.ResearchInfoRAG(self.llm_models, self.retriever, self.web_tool, state, self.app, self.debug).execute()

    def consult_data(self, state: GraphStateType) -> GraphStateType:
        return tool_agents.DataAgent(self.llm_models, self.es_models, state, self.app, self.debug).execute()

    def output_generator(self, state: GraphStateType) -> GraphStateType:
        return main_agents.OutputGenerator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def output_translator(self, state: GraphStateType) -> GraphStateType:
        return flow_agents.OutputTranslator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    # Printers (nodes of the Graph)

    def state_printer(self, state: GraphStateType) -> None:
        return printers.StatePrinter(state, self.debug).execute()

    def final_answer_printer(self, state: GraphStateType) -> None:
        return printers.FinalAnswerPrinter(state, self.debug).execute()
    
    # Routers (conditional edges of the Graph)

    def bypass_router(self, state: GraphStateType) -> str:
        return routers.BypassRouter(state, self.debug).execute()

    def context_router(self, state: GraphStateType) -> str:
        return routers.ContextRouter(state, self.debug).execute()

    def tool_router(self, state: GraphStateType) -> str:
        return routers.ToolRouter(state, self.debug).execute()
    
    def plot_router(self, state: GraphStateType) -> str:
        return routers.InfoTypeRouter(state, self.debug).execute()
    
    def translation_router(self, state: GraphStateType) -> str:
        return routers.TranslationRouter(state, self.debug).execute()
    
    ##### Build the Graph #####

    def build(self) -> CompiledStateGraph:
        workflow = StateGraph(GraphStateType)

        ### Define the nodes ###
        workflow.add_node("input_translator", self.input_translator)
        workflow.add_node("tool_bypasser", self.tool_bypasser)
        workflow.add_node("tool_selector", self.tool_selector)
        workflow.add_node("context_analyzer", self.context_analyzer)
        workflow.add_node("web_search", self.research_info_web) # web search
        workflow.add_node("calculator", self.calculator)
        workflow.add_node("rag_search", self.rag_search)
        workflow.add_node("consult_data", self.consult_data) 
        workflow.add_node("output_generator", self.output_generator)
        workflow.add_node("output_translator", self.output_translator)
        workflow.add_node("context_state_printer", self.state_printer)
        workflow.add_node("final_answer_printer", self.final_answer_printer)

        ### Define the graph topography ###
        
        # Entry and query type routing
        workflow.set_entry_point("input_translator")
        workflow.add_edge("input_translator", "tool_bypasser")
        workflow.add_conditional_edges(
            "tool_bypasser",
            self.bypass_router,
            {
                "output": "output_generator",
                "tool": "tool_selector",
            }
        )
        workflow.add_conditional_edges(
            "tool_selector",
            self.tool_router,
            {
                "web_search": "web_search",
                "calculator": "calculator",
                "rag_search": "rag_search",
                "consult_data": "consult_data",
            }
        )
        
        workflow.add_edge("web_search", "context_analyzer")
        workflow.add_edge("rag_search", "context_analyzer")
        workflow.add_edge("calculator", "context_analyzer")
        workflow.add_edge("consult_data", "context_analyzer")

        workflow.add_conditional_edges(
            "context_analyzer",
            self.context_router,
            {
                "ready_to_answer": "output_generator",
                "need_context": "context_state_printer",
            }
        )

        workflow.add_edge("context_state_printer", "tool_selector")

        # Final steps (generate output and print the final answer)
        workflow.add_conditional_edges(
            "output_generator",
            self.translation_router,
            {
                True: "output_translator",
                False: "final_answer_printer"
            }
        )
        workflow.add_edge("output_translator", "final_answer_printer")
        workflow.add_edge("final_answer_printer", END)
        
        return workflow.compile()
    
    def display_graph(self) -> None:
        compiledGraph = self.build()
        display(Image(compiledGraph.get_graph().draw_mermaid_png()))
        
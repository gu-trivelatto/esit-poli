from abc import ABC
from IPython.display import Image, display

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

import src.libs.agents as agents
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
    
    def date_getter(self, state: GraphStateType) -> GraphStateType:
        return agents.DateGetter(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def input_translator(self, state: GraphStateType) -> GraphStateType:
        return agents.InputTranslator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def tool_bypasser(self, state: GraphStateType) -> GraphStateType:
        return agents.ToolBypasser(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def type_identifier(self, state: GraphStateType) -> GraphStateType:
        return agents.TypeIdentifier(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def mixed(self, state: GraphStateType) -> GraphStateType:
        return agents.Mixed(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def query_generator(self, state: GraphStateType) -> GraphStateType:
        return agents.QueryGenerator(self.llm_models, self.es_models, state, self.app, self.debug).execute()

    def research_info_web(self, state: GraphStateType) -> GraphStateType:
        return agents.ResearchInfoWeb(self.llm_models, retriever, web_tool, state, self.app, self.debug).execute()

    def calculator(self, state: GraphStateType) -> GraphStateType:
        return agents.Calculator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def context_analyzer(self, state: GraphStateType) -> GraphStateType:
        return agents.ContextAnalyzer(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def es_necessary_actions(self, state: GraphStateType) -> GraphStateType:
        return agents.ESNecessaryActionsSelector(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def es_action_selector(self, state: GraphStateType) -> GraphStateType:
        return agents.ESActionSelector(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def input_consolidator(self, state: GraphStateType) -> GraphStateType:
        return agents.InputConsolidator(self.llm_models, self.es_models, state, self.app, self.debug).execute()

    def run_model(self, state: GraphStateType) -> GraphStateType:
        return agents.RunModel(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def modify_model(self, state: GraphStateType) -> GraphStateType:
        return agents.ModifyModel(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def info_type_identifier(self, state: GraphStateType) -> GraphStateType:
        return agents.InfoTypeIdentifier(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def rag_search(self, state: GraphStateType) -> GraphStateType:
        return agents.ResearchInfoRAG(self.llm_models, retriever, web_tool, state, self.app, self.debug).execute()
    
    def consult_model(self, state: GraphStateType) -> GraphStateType:
        return agents.ConsultModel(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def compare_model(self, state: GraphStateType) -> GraphStateType:
        return agents.CompareModel(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def plot_model(self, state: GraphStateType) -> GraphStateType:
        return agents.PlotModel(self.llm_models, self.es_models, state, self.app, self.debug).execute()

    def output_generator(self, state: GraphStateType) -> GraphStateType:
        return agents.OutputGenerator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    def output_translator(self, state: GraphStateType) -> GraphStateType:
        return agents.OutputTranslator(self.llm_models, self.es_models, state, self.app, self.debug).execute()
    
    # Printers (nodes of the Graph)

    def state_printer(self, state: GraphStateType) -> None:
        return printers.StatePrinter(state, self.debug).execute()

    def final_answer_printer(self, state: GraphStateType) -> None:
        return printers.FinalAnswerPrinter(state, self.debug).execute()
    
    # Routers (conditional edges of the Graph)

    def bypass_router(self, state: GraphStateType) -> str:
        return routers.BypassRouter(state, self.debug).execute()

    def type_router(self, state: GraphStateType) -> str:
        return routers.TypeRouter(state, self.debug).execute()

    def mixed_router(self, state: GraphStateType) -> str:
        return routers.MixedRouter(state, self.debug).execute()

    def es_action_router(self, state: GraphStateType) -> str:
        return routers.ESActionRouter(state, self.debug).execute()

    def context_router(self, state: GraphStateType) -> str:
        return routers.ContextRouter(state, self.debug).execute()

    def tool_router(self, state: GraphStateType) -> str:
        return routers.ToolRouter(state, self.debug).execute()
    
    def info_type_router(self, state: GraphStateType) -> str:
        return routers.InfoTypeRouter(state, self.debug).execute()
    
    def translation_router(self, state: GraphStateType) -> str:
        return routers.TranslationRouter(state, self.debug).execute()
    
    ##### Build the Graph #####

    def build(self) -> CompiledStateGraph:
        workflow = StateGraph(GraphStateType)

        ### Define the nodes ###
        workflow.add_node("date_getter", self.date_getter)
        workflow.add_node("input_translator", self.input_translator)
        workflow.add_node("tool_bypasser", self.tool_bypasser)
        workflow.add_node("type_identifier", self.type_identifier)
        workflow.add_node("input_consolidator", self.input_consolidator)
        workflow.add_node("es_necessary_actions", self.es_necessary_actions)
        workflow.add_node("es_action_selector", self.es_action_selector)
        workflow.add_node("context_analyzer", self.context_analyzer)
        workflow.add_node("mixed", self.mixed)
        workflow.add_node("query_generator", self.query_generator)
        workflow.add_node("research_info_web", self.research_info_web) # web search
        workflow.add_node("calculator", self.calculator)
        workflow.add_node("run_model", self.run_model)
        workflow.add_node("modify_model", self.modify_model)
        workflow.add_node("info_type_identifier", self.info_type_identifier)
        workflow.add_node("rag_search", self.rag_search)
        workflow.add_node("consult_model", self.consult_model)
        workflow.add_node("compare_model", self.compare_model)
        workflow.add_node("plot_model", self.plot_model)
        workflow.add_node("output_generator", self.output_generator)
        workflow.add_node("output_translator", self.output_translator)
        workflow.add_node("es_state_printer", self.state_printer)
        workflow.add_node("context_state_printer", self.state_printer)
        workflow.add_node("final_answer_printer", self.final_answer_printer)

        ### Define the graph topography ###
        
        # Entry and query type routing
        workflow.set_entry_point("date_getter")
        workflow.add_edge("date_getter", "input_translator")
        workflow.add_edge("input_translator", "tool_bypasser")
        workflow.add_conditional_edges(
            "tool_bypasser",
            self.bypass_router,
            {
                "output": "output_generator",
                "tool": "type_identifier"
            }
        )
        workflow.add_conditional_edges(
            "type_identifier",
            self.type_router,
            {
                "general": "query_generator",
                "energy_system": "input_consolidator",
                "mixed": "mixed",
            }
        )

        # Mixed query routing
        workflow.add_conditional_edges(
            "mixed",
            self.mixed_router,
            {
                "complete_data": "input_consolidator",
                "needs_data": "query_generator"
            }
        )

        # Energy System branch
        workflow.add_edge("input_consolidator", "es_necessary_actions")
        workflow.add_edge("es_necessary_actions", "es_action_selector")
        workflow.add_conditional_edges(
            "es_action_selector",
            self.es_action_router,
            {
                "run": "run_model",
                "modify": "modify_model",
                "consult": "info_type_identifier",
                "compare": "compare_model",
                "plot": "plot_model",
                "no_action": "output_generator"
            }
        )
        workflow.add_conditional_edges(
            "info_type_identifier",
            self.info_type_router,
            {
                "paper": "rag_search",
                "model": "consult_model",
            }
        )
        workflow.add_edge("run_model", "es_state_printer")
        workflow.add_edge("modify_model", "es_state_printer")
        workflow.add_edge("consult_model", "es_state_printer")
        workflow.add_edge("rag_search", "es_state_printer")
        workflow.add_edge("compare_model", "es_state_printer")
        workflow.add_edge("plot_model", "es_state_printer")
        workflow.add_edge("es_state_printer", "es_action_selector")

        # General branch
        workflow.add_conditional_edges(
            "query_generator",
            self.tool_router,
            {
                "web_search": "research_info_web",
                "calculator": "calculator",
                "direct_output": "output_generator"
            },
        )
        workflow.add_edge("research_info_web", "context_analyzer")
        workflow.add_edge("calculator", "context_analyzer")

        workflow.add_conditional_edges(
            "context_analyzer",
            self.context_router,
            {
                "ready_to_answer": "output_generator",
                "need_context": "context_state_printer",
            }
        )

        workflow.add_conditional_edges(
            "context_state_printer",
            self.type_router,
            {
                "general": "query_generator",
                "mixed": "mixed",
            }
        )

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
        
from abc import ABC

from langchain_community.tools.tavily_search import TavilySearchResults


class WebSearchTool(ABC):
    def __init__(self):
        self.web_search_tool = TavilySearchResults()
    
    def execute(self, query):
        return self.web_search_tool.invoke({"query": query, "max_results": 3})
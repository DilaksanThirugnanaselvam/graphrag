class Summarizer:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def summarize_community(self, community_nodes: list, graph) -> str:
        nodes_str = ", ".join(community_nodes)
        prompt = f"Summarize the community: {nodes_str}"
        return await self.llm_client.generate(prompt)

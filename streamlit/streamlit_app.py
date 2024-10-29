import streamlit as st
import requests
from typing import Optional, Dict, Any, List
from cache_contents import CacheAnalyzer
import os

class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def chat(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": prompt}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with chat API: {str(e)}")
            return None

    def clear_cache(self) -> bool:
        try:
            response = requests.post(f"{self.base_url}/cache/clear")
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Error clearing cache: {str(e)}")
            return False

    def check_health(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}/cache/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting metrics: {str(e)}")
            return None

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'client' not in st.session_state:
        st.session_state.client = ChatClient()
    if 'cache_analyzer' not in st.session_state:
        st.session_state.cache_analyzer = CacheAnalyzer()

def display_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Display a chat message with appropriate styling based on cache status."""
    with st.chat_message(role):
        st.write(content)
        
        if metadata:
            cache_hit = metadata.get('cache_hit', False)
            cache_icon = "🟢" if cache_hit else "⚪"
            st.caption(f"{cache_icon} Cache hit: {cache_hit}")
            if cache_hit:
                st.caption(f"Similarity: {metadata.get('similarity', 0.0):.3f}")

def display_api_status(health: Dict[str, Any]):
    """Display API status with icon and text."""
    if health.get("status") == "healthy":
        st.markdown('<div style="padding: 5px; border-radius: 5px; margin-bottom: 10px;">'
                   '🟢 <span style="color: green;">API Server Healthy</span></div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding: 5px; border-radius: 5px; margin-bottom: 10px;">'
                   '🔴 <span style="color: red;">API Server Disconnected</span></div>',
                   unsafe_allow_html=True)
    st.divider()

def main():
    st.set_page_config(page_title="Semantic Cache Chat", page_icon="💬")
    
    initialize_session_state()
    client = st.session_state.client

    # Sidebar with API status and controls
    with st.sidebar:
        # Display API status at the top of sidebar
        health = client.check_health()
        display_api_status(health)
        
        st.header("Controls")
        if st.button("Clear Cache"):
            if client.clear_cache():
                st.success("Cache cleared successfully")
            else:
                st.error("Failed to clear cache")

        if st.button("Show Metrics"):
            metrics = client.get_metrics()
            if metrics:
                st.subheader("Cache Metrics")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        st.write(f"{key}: {value:.3f}")
                    else:
                        st.write(f"{key}: {value}")
            else:
                st.error("Failed to retrieve metrics")
        
        if st.button("Analyze Cache Contents"):
            with st.spinner("Analyzing cache contents..."):
                analysis = st.session_state.cache_analyzer.summarize_cache()
                if analysis and analysis.get("success"):
                    st.subheader("Cache Analysis")
                    st.write(analysis["summary"])
                    st.caption(f"Total entries: {analysis['total_entries']}")
                else:
                    if analysis and analysis.get("error") == "No entries found in cache":
                        st.info("No entries found in cache. Try chatting first to populate the cache.")
                    else:
                        st.error(analysis.get("error", "Failed to analyze cache contents"))

    st.title("Semantic Cache Chat")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set the OPENAI_API_KEY environment variable to enable cache analysis.")
        if api_key := st.text_input("Or enter your OpenAI API key here:", type="password"):
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set successfully!")
            st.rerun()

    if health.get("status") != "healthy":
        st.error("Error: Chat API is not available")
        return

    # Chat interface
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            metadata=message.get("metadata")
        )

    if prompt := st.chat_input("Enter your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message(role="user", content=prompt)

        # Get bot response
        response = client.chat(prompt)
        if response and 'response' in response:
            message = {
                "role": "assistant",
                "content": response['response'],
                "metadata": {
                    "cache_hit": response.get('cache_hit', False),
                    "similarity": response.get('similarity', 0.0)
                }
            }
            st.session_state.messages.append(message)
            display_message(
                role="assistant",
                content=message["content"],
                metadata=message["metadata"]
            )
        else:
            st.error("Failed to get response from chat API")

if __name__ == "__main__":
    main()

"""
Chat client that interacts with the semantic cache chat API.
"""

import requests
import sys
from typing import Optional, Dict, Any, List

class ChatClient:
    """
    Client for interacting with the semantic cache chat API.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):  # Updated to port 8004
        """
        Initialize the chat client.

        Args:
            base_url: Base URL of the chat API
        """
        self.base_url = base_url.rstrip('/')

    def chat(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Send a chat prompt to the API and get the response.

        Args:
            prompt: The user's prompt

        Returns:
            Dictionary containing response and cache information if successful, None otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": prompt}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with chat API: {str(e)}")
            return None

    def clear_cache(self) -> bool:
        """
        Clear the semantic cache.

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(f"{self.base_url}/cache/clear")
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error clearing cache: {str(e)}")
            return False

    def check_health(self) -> Dict[str, Any]:
        """
        Check if the API is healthy.

        Returns:
            Dict containing health status information
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error checking API health: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    def analyze_cache(self) -> Optional[Dict[str, Any]]:
        """
        Get cache analysis results.

        Returns:
            Dictionary containing analysis results if successful, None otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/cache/analyze")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error analyzing cache: {str(e)}")
            return None

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics if successful, None otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/cache/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting cache stats: {str(e)}")
            return None

    def batch_chat(self, prompts: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Send multiple chat prompts to the API in batch.

        Args:
            prompts: List of user prompts

        Returns:
            List of response dictionaries if successful, None otherwise
        """
        # Process queries sequentially since there's no batch endpoint
        try:
            responses = []
            for prompt in prompts:
                response = self.chat(prompt)
                if response:
                    responses.append(response)
                else:
                    return None
            return responses
        except Exception as e:
            print(f"Error processing batch requests: {str(e)}")
            return None

def main():
    """
    Main function to run the interactive chat client.
    """
    client = ChatClient()

    # Check if API is healthy before starting
    health = client.check_health()
    if health.get("status") != "healthy":
        print("Error: Chat API is not available")
        sys.exit(1)

    print("Welcome to the Interactive Chat with Semantic Cache")
    print("Type 'quit' to end the conversation.")
    print("Type 'clear cache' to clear the entire cache for this session.")
    print("Type 'analyze' to view cache analysis.")
    print("Type 'stats' to view cache statistics.")
    print("Type 'batch' to enter batch mode for multiple queries.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear cache':
            if client.clear_cache():
                print("Cache cleared successfully.")
            else:
                print("Failed to clear cache.")
            continue
        elif user_input.lower() == 'analyze':
            analysis = client.analyze_cache()
            if analysis:
                print("\nCache Analysis:")
                for key, value in analysis.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.3f}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("Failed to retrieve cache analysis.")
            continue
        elif user_input.lower() == 'stats':
            stats = client.get_stats()
            if stats:
                print("\nCache Statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.3f}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("Failed to retrieve cache statistics.")
            continue
        elif user_input.lower() == 'batch':
            print("Enter your queries (one per line). Type 'end' on a new line when done:")
            queries = []
            while True:
                query = input()
                if query.lower() == 'end':
                    break
                queries.append(query)
            
            if queries:
                responses = client.batch_chat(queries)
                if responses:
                    print("\nResponses:")
                    for i, (query, response) in enumerate(zip(queries, responses), 1):
                        print(f"\nQuery {i}: {query}")
                        print(f"Response: {response.get('response', 'No response')}")
                        print(f"Cache hit: {response.get('cache_hit', False)}")
                        if response.get('cache_hit'):
                            print(f"Similarity: {response.get('similarity', 0.0):.3f}")
                else:
                    print("Failed to get batch responses from chat API.")
            continue

        response = client.chat(user_input)
        if response:
            if 'response' in response:
                print(f"Assistant: {response['response']}")
            print(f"Cache hit: {response.get('cache_hit', False)}")
            if response.get('cache_hit'):
                print(f"Similarity: {response.get('similarity', 0.0):.3f}")
        else:
            print("Failed to get response from chat API.")
            break

if __name__ == "__main__":
    main()

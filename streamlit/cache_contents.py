import os
import requests
from typing import Optional, Dict, Any, List

class CacheAnalyzer:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def get_cache_contents(self) -> Optional[List[Dict[str, Any]]]:
        """Get all entries from the semantic cache."""
        try:
            response = requests.get(f"{self.base_url}/cache/contents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting cache contents: {str(e)}")
            return None

    def summarize_cache(self) -> Optional[Dict[str, Any]]:
        """Analyze and summarize the cache contents using the API."""
        try:
            response = requests.get(f"{self.base_url}/cache/analyze")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error analyzing cache: {str(e)}",
                "success": False
            }

if __name__ == "__main__":
    # Create an instance of CacheAnalyzer
    analyzer = CacheAnalyzer()

    # Get and print the summary
    summary = analyzer.summarize_cache()

    if summary.get("success"):
        print("\nCache Analysis Summary:")
        print("=" * 50)
        print(summary["summary"])
        print("\nTotal Entries:", summary["total_entries"])
    else:
        print("\nError:", summary.get("error"))

from typing import List, Dict, Any, Optional
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from models.exceptions import DatabaseError, EmbeddingError
from models.cache_metrics import CacheMetrics
import logging
import time
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class OpenAIService:
    def __init__(self, api_key: str, model: str, metrics: CacheMetrics):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.metrics = metrics

    def get_completion(self, query: str) -> str:
        """Get a single completion from OpenAI API"""
        with tracer.start_as_current_span("openai_api_call"):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                if isinstance(e, RateLimitError):
                    self.metrics.record_error("api")
                    logger.error("OpenAI API rate limit exceeded")
                elif isinstance(e, APIConnectionError):
                    self.metrics.record_error("api")
                    logger.error("Failed to connect to OpenAI API")
                elif isinstance(e, APIError):
                    self.metrics.record_error("api")
                    logger.error(f"OpenAI API error: {str(e)}")
                raise

    def get_batch_completions(self, queries: List[str]) -> List[str]:
        """Get multiple completions from OpenAI API in batch"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."}
                ] + [{"role": "user", "content": query} for query in queries]
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            if isinstance(e, (RateLimitError, APIConnectionError, APIError)):
                self.metrics.record_error("api")
                logger.error(f"OpenAI API error: {str(e)}")
            raise

    def analyze_cache_contents(self, contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cache contents using OpenAI."""
        if not contents:
            logger.info("No entries found in cache")
            return {
                "success": False,
                "error": "No entries found in cache",
                "total_entries": 0
            }

        # Prepare the data for analysis
        queries = [entry.get("query", "") for entry in contents]
        responses = [entry.get("response", "") for entry in contents]

        # Create a prompt for OpenAI
        prompt = f"""Analyze the following cache of questions and answers:

Questions:
{chr(10).join(f'- {q}' for q in queries)}

Answers:
{chr(10).join(f'- {r}' for r in responses)}

Please provide:
1. A brief summary of the main topics discussed
2. The types of questions being asked
3. Any patterns in the responses
4. Total number of entries: {len(contents)}

Format the response in a clear, bulleted structure."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing cache contents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return {
                "summary": response.choices[0].message.content,
                "total_entries": len(contents),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error analyzing cache contents: {str(e)}")
            if isinstance(e, (RateLimitError, APIConnectionError, APIError)):
                self.metrics.record_error("api")
            return {
                "error": f"Error generating summary: {str(e)}",
                "total_entries": len(contents),
                "success": False
            }

    def get_current_model(self) -> str:
        """Get the current OpenAI model being used."""
        return self.model

    def set_model(self, model: str) -> bool:
        """Set the OpenAI model to use for completions."""
        try:
            # Test the model with a simple completion to verify it works
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test message"}
                ],
                max_tokens=10
            )
            # If successful, update the model
            self.model = model
            logger.info(f"Successfully updated model to: {model}")
            return True
        except Exception as e:
            logger.error(f"Error setting model to {model}: {str(e)}")
            raise

import asyncio
import logging
import logging.config
import json
import os
import time
import sys
from typing import List, Dict
import aiohttp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.settings import (
    LOGGING_CONFIG, 
    CACHE_SIMILARITY_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL
)
from services.cache_service import CacheService
from services.embedding_service import EmbeddingService
from services.openai_service import OpenAIService
from models.cache_metrics import CacheMetrics

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('stress_test')

async def send_request(session: aiohttp.ClientSession, query: str) -> Dict:
    """Send a single request to the API."""
    try:
        async with session.post('http://localhost:8000/query', 
                              json={'query': query}) as response:
            return await response.json()
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return {'error': str(e)}

async def run_test(queries: List[str], concurrency: int = 10) -> List[Dict]:
    """Run the stress test with the given queries and concurrency level."""
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async with aiohttp.ClientSession() as session:
        async def bounded_request(query: str) -> Dict:
            async with semaphore:
                return await send_request(session, query)
        
        tasks = [bounded_request(query) for query in queries]
        for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await completed
            results.append(result)
    
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze the test results."""
    total_requests = len(results)
    cache_hits = sum(1 for r in results if r.get('cache_hit', False))
    cache_misses = total_requests - cache_hits
    
    latencies = [r.get('latency', 0) for r in results if 'latency' in r]
    if latencies:
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)  # Added p99 calculation
    else:
        avg_latency = min_latency = max_latency = p95_latency = p99_latency = 0
    
    errors = [r for r in results if 'error' in r]
    
    analysis = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_requests': total_requests,
        'successful_requests': total_requests - len(errors),
        'failed_requests': len(errors),
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'hit_rate': cache_hits / total_requests if total_requests > 0 else 0,
        'latency': {
            'average': avg_latency,
            'min': min_latency,
            'max': max_latency,
            'p95': p95_latency,
            'p99': p99_latency  # Added p99 to output
        }
    }
    
    return analysis

def save_results(analysis: Dict, results: List[Dict]):
    """Save the test results to a file."""
    output = {
        'analysis': analysis,
        'raw_results': results
    }
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Save to logs/stress_test_results.json
    output_path = os.path.join('logs', 'stress_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Log analysis results
    logger.info("Test Analysis:")
    logger.info(f"Total Requests: {analysis['total_requests']}")
    logger.info(f"Successful Requests: {analysis['successful_requests']}")
    logger.info(f"Failed Requests: {analysis['failed_requests']}")
    logger.info(f"Cache Hit Rate: {analysis['hit_rate']:.2%}")
    logger.info(f"Average Latency: {analysis['latency']['average']:.3f}s")
    logger.info(f"P95 Latency: {analysis['latency']['p95']:.3f}s")
    logger.info(f"P99 Latency: {analysis['latency']['p99']:.3f}s")  # Added p99 to logs

def visualize_results(analysis: Dict):
    """Generate visualization of the test results."""
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Stress Test Results Analysis', fontsize=16, y=0.95)

    # 1. Success vs Failure Bar Chart (Top Left)
    ax1 = plt.subplot(221)
    success_fail = [analysis['successful_requests'], analysis['failed_requests']]
    ax1.bar(['Successful', 'Failed'], success_fail, color=['green', 'red'])
    ax1.set_title('Request Results')
    ax1.set_ylabel('Number of Requests')
    for i, v in enumerate(success_fail):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    # 2. Cache Hits vs Misses Pie Chart (Top Right)
    ax2 = plt.subplot(222)
    cache_labels = ['Cache Hits', 'Cache Misses']
    cache_values = [analysis['cache_hits'], analysis['cache_misses']]
    ax2.pie(cache_values, labels=cache_labels, autopct='%1.1f%%', colors=['lightblue', 'orange'])
    ax2.set_title('Cache Performance')

    # 3. Response Time Metrics (Bottom Left)
    ax3 = plt.subplot(223)
    latency = analysis['latency']
    metrics = ['min', 'average', 'p95', 'p99', 'max']
    values = [latency['min'], latency['average'], latency['p95'], latency['p99'], latency['max']]
    ax3.bar(metrics, values, color='skyblue')
    ax3.set_title('Latency Metrics')
    ax3.set_ylabel('Seconds')
    plt.xticks(rotation=45)

    # 4. Success and Cache Hit Rates (Bottom Right)
    ax4 = plt.subplot(224)
    success_rate = (analysis['successful_requests'] / analysis['total_requests']) * 100
    hit_rate = analysis['hit_rate'] * 100
    rates = ['Success Rate', 'Cache Hit Rate']
    rate_values = [success_rate, hit_rate]
    ax4.bar(rates, rate_values, color=['lightgreen', 'orange'])
    ax4.set_title('Performance Rates')
    ax4.set_ylabel('Percentage')
    for i, v in enumerate(rate_values):
        ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    ax4.set_ylim(0, 100)

    # Add overall stats as text
    plt.figtext(0.02, 0.02, 
        f"Total Requests: {analysis['total_requests']}\n"
        f"Timestamp: {analysis['timestamp']}",
        fontsize=10, ha='left')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('stress_test_visualization.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved as stress_test_visualization.png")

async def main():
    # Prompt user about cache clearing
    print("\nWARNING: This stress test will clear the existing cache.")
    response = input("Do you want to proceed? (yes/no): ").lower().strip()
    
    if response != 'yes':
        print("Stress test cancelled.")
        sys.exit(0)
    
    # Initialize metrics
    metrics = CacheMetrics()
    
    # Initialize services with required parameters
    embedding_service = EmbeddingService(metrics=metrics)
    openai_service = OpenAIService(
        api_key=OPENAI_API_KEY,
        model=OPENAI_CHAT_MODEL,
        metrics=metrics
    )
    
    # Initialize cache service
    cache_service = CacheService(
        db_path="cache.db",
        collection_name="semantic_cache",
        similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
        cache_ttl=3600,  # 1 hour TTL
        metrics=metrics,
        embedding_service=embedding_service,
        openai_service=openai_service
    )
    
    # Clear the cache
    logger.info("Clearing cache...")
    cache_service.clear_cache()  # Removed await since it's not an async method
    logger.info("Cache cleared successfully.")

    # Example queries
    queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "What is the meaning of life?",
        "How tall is Mount Everest?",
    ] * 20  # Repeat queries to test caching

    logger.info("Starting stress test...")
    logger.info(f"Number of queries: {len(queries)}")
    
    # Run the test
    results = await run_test(queries)
    
    # Analyze and save results
    analysis = analyze_results(results)
    save_results(analysis, results)
    
    # Generate visualization
    visualize_results(analysis)
    
    logger.info("Stress test completed.")

if __name__ == "__main__":
    asyncio.run(main())

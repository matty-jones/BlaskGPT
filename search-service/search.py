"""
Web Search Service

Provides web search functionality with preference for concise answers.
Uses DuckDuckGo by default (no API key required, privacy-focused).
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try new package name first, fallback to old name
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("Neither 'ddgs' nor 'duckduckgo_search' package is installed")


class SearchService:
    """Service for performing web searches"""
    
    def __init__(self, provider: str = "duckduckgo"):
        self.provider = provider
        self._ddgs: Optional[DDGS] = None
    
    def _get_ddgs(self) -> DDGS:
        """Get or create DuckDuckGo search instance"""
        if self._ddgs is None:
            self._ddgs = DDGS()
        return self._ddgs
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "us-en"
    ) -> List[Dict[str, str]]:
        """
        Perform a web search with retry logic for rate limiting
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Search region (default: us-en)
        
        Returns:
            List of search results with title, snippet, and url
        """
        import asyncio
        # Try new package exception first, fallback to old
        try:
            from ddgs.exceptions import DuckDuckGoSearchException
        except ImportError:
            try:
                from duckduckgo_search.exceptions import DuckDuckGoSearchException
            except ImportError:
                # Fallback: create a generic exception class
                class DuckDuckGoSearchException(Exception):
                    pass
        
        max_retries = 3
        retry_delay = 5  # seconds - start with longer delay for rate limits
        
        for attempt in range(max_retries):
            try:
                ddgs = self._get_ddgs()
                results = []
                
                # Run search in executor to avoid blocking
                loop = asyncio.get_event_loop()
                search_results = await loop.run_in_executor(
                    None,
                    lambda: list(ddgs.text(query, max_results=max_results, region=region))
                )
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", "")
                    })
                
                logger.info(f"Search for '{query}' returned {len(results)} results")
                return results
            
            except DuckDuckGoSearchException as e:
                error_msg = str(e).lower()
                if "ratelimit" in error_msg or "rate" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        # Create a new DDGS instance for retry
                        self._ddgs = None
                        continue
                    else:
                        logger.error(f"Search rate limited after {max_retries} attempts: {e}")
                        return []
                else:
                    logger.error(f"Search failed with DuckDuckGo error: {e}", exc_info=True)
                    return []
            
            except Exception as e:
                logger.error(f"Search failed: {e}", exc_info=True)
                return []
        
        # If we exhausted all retries without returning
        return []
    
    def format_search_context(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results into context for LLM
        
        Args:
            results: List of search result dictionaries
        
        Returns:
            Formatted string with search context
        """
        if not results:
            return "No search results found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            
            context_parts.append(
                f"[{i}] {title}\n{snippet}\nSource: {url}\n"
            )
        
        return "\n".join(context_parts)


# Global instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create global search service instance"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

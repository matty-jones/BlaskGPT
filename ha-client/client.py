"""
Home Assistant API Client

Handles authentication, entity discovery, and service calls to Home Assistant.
"""

import logging
from typing import Dict, List, Optional, Any
import aiohttp
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """HA Client settings"""
    ha_url: str
    ha_access_token: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class HomeAssistantClient:
    """Client for interacting with Home Assistant API"""
    
    def __init__(self, url: str, access_token: str):
        self.url = url.rstrip('/')
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to Home Assistant API"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        
        url = f"{self.url}/api/{endpoint}"
        try:
            async with self._session.request(
                method, url, headers=self.headers, **kwargs
            ) as response:
                if response.status >= 400:
                    # Log error response body for debugging
                    try:
                        error_body = await response.text()
                        logger.error(f"HA API request failed {response.status}: {error_body[:500]}")
                    except:
                        pass
                response.raise_for_status()
                if response.content_type == 'application/json':
                    return await response.json()
                return {"status": "ok", "text": await response.text()}
        except aiohttp.ClientError as e:
            logger.error(f"HA API request failed: {e}")
            raise
    
    async def get_config(self) -> Dict[str, Any]:
        """Get Home Assistant configuration"""
        return await self._request("GET", "config")
    
    async def get_entities(self) -> List[Dict[str, Any]]:
        """Get all entities from Home Assistant"""
        return await self._request("GET", "states")
    
    async def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get state of a specific entity"""
        return await self._request("GET", f"states/{entity_id}")
    
    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        return_response: bool = False,
        **service_data
    ) -> List[Dict[str, Any]]:
        """
        Call a Home Assistant service
        
        Args:
            domain: Service domain (e.g., 'media_player', 'shopping_list')
            service: Service name (e.g., 'play_media', 'add_item')
            entity_id: Optional entity ID to target
            return_response: If True, add ?return_response=true to URL (for services that return data)
            **service_data: Additional service data
        """
        data = service_data.copy()
        if entity_id:
            data["entity_id"] = entity_id
        
        endpoint = f"services/{domain}/{service}"
        if return_response:
            endpoint += "?return_response=true"
        
        return await self._request(
            "POST",
            endpoint,
            json=data
        )
    
    async def search_entities(
        self,
        query: str,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name or attributes
        
        Args:
            query: Search query
            domain: Optional domain filter (e.g., 'media_player')
        """
        entities = await self.get_entities()
        results = []
        query_lower = query.lower()
        
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            friendly_name = entity.get("attributes", {}).get("friendly_name", "")
            
            # Filter by domain if specified
            if domain and not entity_id.startswith(f"{domain}."):
                continue
            
            # Search in entity_id and friendly_name
            if (query_lower in entity_id.lower() or 
                query_lower in friendly_name.lower()):
                results.append(entity)
        
        return results
    
    async def find_speakers(self) -> List[Dict[str, Any]]:
        """Find all media player entities (speakers)"""
        return await self.search_entities("", domain="media_player")
    
    async def test_connection(self) -> bool:
        """Test connection to Home Assistant"""
        try:
            await self.get_config()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Convenience functions for common operations
async def play_media_on_speakers(
    client: HomeAssistantClient,
    entity_ids: List[str],
    media_content_id: str,
    media_content_type: str = "music"
) -> Dict[str, Any]:
    """Play media on specified speakers"""
    result = await client.call_service(
        "media_player",
        "play_media",
        entity_id=entity_ids,
        media_content_id=media_content_id,
        media_content_type=media_content_type
    )
    return result


async def add_to_shopping_list(
    client: HomeAssistantClient,
    item: str
) -> Dict[str, Any]:
    """Add item to shopping list"""
    result = await client.call_service(
        "shopping_list",
        "add_item",
        name=item
    )
    return result


async def start_timer(
    client: HomeAssistantClient,
    duration_minutes: int,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """Start a timer"""
    duration_seconds = duration_minutes * 60
    service_data = {"duration": duration_seconds}
    if name:
        service_data["name"] = name
    
    result = await client.call_service(
        "timer",
        "start",
        **service_data
    )
    return result

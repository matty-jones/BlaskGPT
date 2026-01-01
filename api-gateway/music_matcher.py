"""
Fuzzy music matching and selection logic.

Provides scoring and selection functions for matching user queries to music search results.
"""
import re
import logging
from typing import Optional, List, Dict, Any
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Thresholds for hierarchy-based selection
TRACK_STRONG = 90
PLAYLIST_STRONG = 85
ARTIST_STRONG = 80
FALLBACK_MIN = 60


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def score_candidate(
    query: str,
    candidate_name: str,
    candidate_type: str,
    artist_hint: Optional[str] = None,
    candidate_artist: Optional[str] = None,
) -> float:
    """
    Score a candidate match using fuzzy matching and bonuses.
    
    Args:
        query: The search query
        candidate_name: Name of the candidate (track, artist, playlist, album)
        candidate_type: Type of candidate ("track", "artist", "playlist", "album")
        artist_hint: Optional artist name from user query (e.g., "by Rush")
        candidate_artist: Optional artist name of the candidate
        
    Returns:
        Score between 0-115 (base 0-100 + bonuses up to +15)
    """
    query_norm = normalize_text(query)
    candidate_norm = normalize_text(candidate_name)
    
    # Base score: max of three rapidfuzz ratios
    ratio_score = fuzz.ratio(query_norm, candidate_norm)
    partial_score = fuzz.partial_ratio(query_norm, candidate_norm)
    token_score = fuzz.token_sort_ratio(query_norm, candidate_norm)
    
    base_score = max(ratio_score, partial_score, token_score)
    
    # Bonuses
    bonuses = 0
    
    # Exact match bonus
    if query_norm == candidate_norm:
        bonuses += 15
        logger.debug(f"Exact match bonus: {candidate_name}")
    
    # Token start/end bonus
    query_tokens = query_norm.split()
    candidate_tokens = candidate_norm.split()
    if query_tokens and candidate_tokens:
        if query_tokens[0] == candidate_tokens[0] or query_tokens[-1] == candidate_tokens[-1]:
            bonuses += 5
            logger.debug(f"Token match bonus: {candidate_name}")
    
    # Artist hint bonus
    if artist_hint and candidate_artist:
        artist_hint_norm = normalize_text(artist_hint)
        candidate_artist_norm = normalize_text(candidate_artist)
        if artist_hint_norm in candidate_artist_norm or candidate_artist_norm in artist_hint_norm:
            # Use fuzzy matching for artist comparison
            artist_ratio = fuzz.ratio(artist_hint_norm, candidate_artist_norm)
            if artist_ratio >= 80:  # Strong artist match
                bonuses += 10
                logger.debug(f"Artist hint bonus: {candidate_name} by {candidate_artist}")
    
    final_score = base_score + bonuses
    logger.debug(f"Scored '{candidate_name}' ({candidate_type}): base={base_score}, bonuses={bonuses}, final={final_score}")
    
    return final_score


def extract_type_hint(raw_utterance: str) -> Optional[str]:
    """
    Extract media type hint from user utterance using keyword matching.
    
    Args:
        raw_utterance: Original user message
        
    Returns:
        Type hint ("playlist", "artist", "track", "album") or None
    """
    utterance_lower = raw_utterance.lower()
    
    # Check for explicit type keywords
    if "playlist" in utterance_lower:
        return "playlist"
    elif "artist" in utterance_lower:
        return "artist"
    elif "track" in utterance_lower or "song" in utterance_lower:
        return "track"
    elif "album" in utterance_lower:
        return "album"
    
    return None


def extract_artist_hint(raw_utterance: str) -> Optional[str]:
    """
    Extract artist hint from user utterance using simple regex.
    
    Looks for patterns like "by <artist>" or "artist <artist>".
    
    Args:
        raw_utterance: Original user message
        
    Returns:
        Artist name or None
    """
    # Pattern 1: "by <artist>" or "by the <artist>"
    match = re.search(r'\bby\s+(?:the\s+)?([^,\.!?]+)', raw_utterance, re.IGNORECASE)
    if match:
        artist = match.group(1).strip()
        # Remove trailing words that might be part of the sentence
        artist = re.sub(r'\s+(on|in|at|from|to|the|a|an)\s+.*$', '', artist, flags=re.IGNORECASE)
        if artist:
            return artist
    
    # Pattern 2: "artist <artist>" (less common but possible)
    match = re.search(r'\bartist\s+([^,\.!?]+)', raw_utterance, re.IGNORECASE)
    if match:
        artist = match.group(1).strip()
        if artist:
            return artist
    
    return None


def select_best_match(
    query: str,
    search_results: List[Dict[str, Any]],
    type_hint: Optional[str] = None,
    artist_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select the best match from search results using fuzzy scoring and hierarchy.
    
    Args:
        query: The search query
        search_results: List of search result dicts with keys like uri, name, media_type, artists
        type_hint: Optional type hint ("playlist", "artist", "track", "album")
        artist_hint: Optional artist name hint
        
    Returns:
        Best match dict with uri, media_type, name, etc., or None if no good match
    """
    if not search_results:
        return None
    
    # Score all candidates
    scored_candidates = []
    for item in search_results:
        if not isinstance(item, dict):
            continue
        
        uri = item.get("uri", item.get("media_content_id", ""))
        name = item.get("name", item.get("title", "Unknown"))
        item_type = item.get("media_type", item.get("type", "music"))
        
        # Extract artist from item
        candidate_artist = None
        artists_list = item.get("artists", [])
        if isinstance(artists_list, list) and len(artists_list) > 0:
            if isinstance(artists_list[0], dict):
                candidate_artist = artists_list[0].get("name", "")
            else:
                candidate_artist = str(artists_list[0])
        else:
            candidate_artist = item.get("artist", item.get("artist_name", ""))
        
        score = score_candidate(
            query=query,
            candidate_name=name,
            candidate_type=item_type,
            artist_hint=artist_hint,
            candidate_artist=candidate_artist,
        )
        
        scored_candidates.append({
            "item": item,
            "score": score,
            "uri": uri,
            "name": name,
            "type": item_type,
            "artist": candidate_artist,
        })
    
    if not scored_candidates:
        return None
    
    # Group by type and find best in each category
    best_by_type = {}
    for candidate in scored_candidates:
        item_type = candidate["type"]
        if item_type not in best_by_type or candidate["score"] > best_by_type[item_type]["score"]:
            best_by_type[item_type] = candidate
    
    logger.info(f"[MusicMatcher] Best scores by type: {[(t, b['score']) for t, b in best_by_type.items()]}")
    
    # Apply hierarchy logic
    best_track = best_by_type.get("track")
    best_playlist = best_by_type.get("playlist")
    best_artist = best_by_type.get("artist")
    best_album = best_by_type.get("album")
    
    # If type hint provided, prioritize that type
    if type_hint:
        if type_hint == "track" and best_track and best_track["score"] >= 70:
            logger.info(f"[MusicMatcher] Type hint 'track' matched: {best_track['name']} (score: {best_track['score']})")
            return {
                "uri": best_track["uri"],
                "media_type": "track",
                "name": best_track["name"],
                "artist": best_track["artist"],
                "score": best_track["score"],
            }
        elif type_hint == "playlist" and best_playlist and best_playlist["score"] >= 70:
            logger.info(f"[MusicMatcher] Type hint 'playlist' matched: {best_playlist['name']} (score: {best_playlist['score']})")
            return {
                "uri": best_playlist["uri"],
                "media_type": "playlist",
                "name": best_playlist["name"],
                "score": best_playlist["score"],
            }
        elif type_hint == "artist" and best_artist and best_artist["score"] >= 70:
            logger.info(f"[MusicMatcher] Type hint 'artist' matched: {best_artist['name']} (score: {best_artist['score']})")
            return {
                "uri": best_artist["uri"],
                "media_type": "artist",
                "name": best_artist["name"],
                "score": best_artist["score"],
            }
        elif type_hint == "album" and best_album and best_album["score"] >= 70:
            logger.info(f"[MusicMatcher] Type hint 'album' matched: {best_album['name']} (score: {best_album['score']})")
            return {
                "uri": best_album["uri"],
                "media_type": "album",
                "name": best_album["name"],
                "score": best_album["score"],
            }
    
    # No explicit hint or hint didn't match → use hierarchy
    if best_track and best_track["score"] >= TRACK_STRONG:
        logger.info(f"[MusicMatcher] Strong track match: {best_track['name']} (score: {best_track['score']})")
        return {
            "uri": best_track["uri"],
            "media_type": "track",
            "name": best_track["name"],
            "artist": best_track["artist"],
            "score": best_track["score"],
        }
    elif best_playlist and best_playlist["score"] >= PLAYLIST_STRONG:
        logger.info(f"[MusicMatcher] Strong playlist match: {best_playlist['name']} (score: {best_playlist['score']})")
        return {
            "uri": best_playlist["uri"],
            "media_type": "playlist",
            "name": best_playlist["name"],
            "score": best_playlist["score"],
        }
    elif best_artist and best_artist["score"] >= ARTIST_STRONG:
        logger.info(f"[MusicMatcher] Strong artist match: {best_artist['name']} (score: {best_artist['score']})")
        return {
            "uri": best_artist["uri"],
            "media_type": "artist",
            "name": best_artist["name"],
            "score": best_artist["score"],
        }
    else:
        # Fallback: best "something" if it's at least plausible
        all_candidates = [best_track, best_playlist, best_artist, best_album]
        all_candidates = [c for c in all_candidates if c is not None]
        
        if all_candidates:
            best_fallback = max(all_candidates, key=lambda c: c["score"])
            if best_fallback["score"] >= FALLBACK_MIN:
                logger.info(f"[MusicMatcher] Fallback match: {best_fallback['name']} ({best_fallback['type']}, score: {best_fallback['score']})")
                return {
                    "uri": best_fallback["uri"],
                    "media_type": best_fallback["type"],
                    "name": best_fallback["name"],
                    "artist": best_fallback.get("artist"),
                    "score": best_fallback["score"],
                }
    
    # Everything is rubbish
    logger.warning(f"[MusicMatcher] No good match found. Best score: {max([c['score'] for c in scored_candidates]) if scored_candidates else 0}")
    return None





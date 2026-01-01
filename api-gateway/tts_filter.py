"""
TTS Filter Module

Filters URLs and emojis from text to make it TTS-friendly.
Designed to work with both non-streaming and streaming responses.
"""

import re
from typing import Optional


def _normalize_punctuation_spacing(text: str) -> str:
    """
    Normalize punctuation spacing for better TTS prosody.
    
    Module-level function for use inside and outside the class.
    
    - Ensures space after sentence-ending punctuation (., !, ?) when followed by a letter
    - Ensures space after colons and semicolons when followed by a letter
    - Normalizes all whitespace (newlines, tabs, multiple spaces) to single spaces
    
    Args:
        text: Text to normalize
        
    Returns:
        Text with normalized punctuation spacing
    """
    if not text:
        return ""
    
    # First, normalize all whitespace (newlines, tabs, multiple spaces) to single spaces
    text = re.sub(r'[\s\n\r\t]+', ' ', text)
    
    # Ensure space after sentence-ending punctuation (., !, ?) when followed by a letter
    # Pattern: punctuation followed by a letter (no space in between)
    text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text)
    
    # Ensure space after colons and semicolons when followed by a letter
    # Pattern: colon/semicolon followed by a letter (no space in between)
    text = re.sub(r'([:;])([a-zA-Z])', r'\1 \2', text)
    
    # Clean up any multiple spaces that might have been created
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


class StreamingTTSFilter:
    """
    Stateful filter for streaming text that removes URLs and emojis incrementally.
    Handles partial URLs/emojis that may be split across chunks.
    """
    
    def __init__(self):
        # Buffer to hold partial text that might be part of a URL or emoji
        self.buffer = ""
        # Maximum buffer size to prevent memory issues (URLs are typically < 200 chars)
        self.max_buffer_size = 200
        
        # URL start pattern (for detecting URL beginnings in streaming)
        # Matches: http://, https://, www., IP addresses, domains
        self.url_start_pattern = re.compile(
            r'(https?://|www\.|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[a-zA-Z0-9-]+\.[a-zA-Z]{2,})'
        )
        
        # Full URL pattern (for removing complete URLs)
        # Matches: http://..., https://..., www.example.com, 192.168.1.1:8080, example.com/path, etc.
        self.url_pattern = re.compile(
            r'(https?://[^\s<>"\'\)]+|www\.[^\s<>"\'\)]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s<>"\'\)]*|'
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:/[^\s<>"\'\)]*)?)'
        )
        
        # Emoji pattern (Unicode ranges)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE
        )
    
    def filter_chunk(self, chunk: str) -> str:
        """
        Filter a chunk of text, handling partial URLs/emojis at boundaries.
        Returns sanitized text that's safe to send to TTS.
        """
        if not chunk:
            return ""
        
        # Add chunk to buffer
        self.buffer += chunk
        
        # If buffer is too large, process it in chunks to avoid memory issues
        if len(self.buffer) > self.max_buffer_size:
            # Process up to max_buffer_size, keep remainder in buffer
            to_process = self.buffer[:self.max_buffer_size]
            self.buffer = self.buffer[self.max_buffer_size:]
            return self._process_buffer(to_process)
        
        # Check if we might have a complete URL or emoji
        # URLs typically end with whitespace, punctuation, or end of string
        # We'll be conservative and only emit text when we're confident it's safe
        
        # Look for URL start patterns in the buffer
        url_match = self.url_start_pattern.search(self.buffer)
        if url_match:
            # Found URL start - check if we have a complete URL
            url_start = url_match.start()
            # Look for URL end (whitespace, punctuation, or end of buffer)
            url_end_match = re.search(r'[\s\n\r\t.,;!?)]', self.buffer[url_match.end():])
            if url_end_match:
                # Complete URL found - remove it
                url_end = url_match.end() + url_end_match.start()
                self.buffer = self.buffer[:url_start] + self.buffer[url_end:]
            else:
                # Incomplete URL - keep in buffer, don't emit yet
                # Emit text before URL start
                safe_text = self.buffer[:url_start]
                self.buffer = self.buffer[url_start:]
                safe_text = self._process_emojis(safe_text)
                return self._normalize_punctuation_spacing(safe_text)
        
        # Check for emojis (these are usually complete in a single chunk)
        if self.emoji_pattern.search(self.buffer):
            # Remove emojis from buffer
            self.buffer = self.emoji_pattern.sub('', self.buffer)
        
        # If buffer doesn't contain URL start patterns, we can emit it
        # But be conservative: only emit if we have whitespace or punctuation
        # that suggests we're not in the middle of a URL
        if not self.url_start_pattern.search(self.buffer):
            # No URL pattern - safe to emit after emoji removal
            result = self._process_emojis(self.buffer)
            self.buffer = ""
            return self._normalize_punctuation_spacing(result)
        
        # Still might be building a URL - only emit if we have clear word boundaries
        # Look for the last safe break point (whitespace before any URL pattern)
        last_space = self.buffer.rfind(' ')
        if last_space > 0:
            url_in_remainder = self.url_start_pattern.search(self.buffer[last_space:])
            if not url_in_remainder:
                # Safe to emit up to last space
                safe_text = self.buffer[:last_space + 1]
                self.buffer = self.buffer[last_space + 1:]
                safe_text = self._process_emojis(safe_text)
                return self._normalize_punctuation_spacing(safe_text)
        
        # Unsure - keep in buffer
        return ""
    
    def flush(self) -> str:
        """
        Flush remaining buffer content. Call this when stream ends.
        Removes any remaining URLs/emojis from the buffer and normalizes punctuation.
        """
        if not self.buffer:
            return ""
        
        # Remove any URLs (even incomplete ones)
        self.buffer = self.url_pattern.sub('', self.buffer)
        # Remove emojis
        result = self._process_emojis(self.buffer)
        # Normalize punctuation spacing for TTS prosody
        result = self._normalize_punctuation_spacing(result)
        self.buffer = ""
        return result
    
    def _process_buffer(self, text: str) -> str:
        """Process a buffer chunk: remove URLs and emojis, normalize punctuation."""
        # Remove URLs
        text = self.url_pattern.sub('', text)
        # Remove emojis
        text = self._process_emojis(text)
        # Normalize punctuation spacing for TTS prosody
        text = self._normalize_punctuation_spacing(text)
        return text
    
    def _process_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return self.emoji_pattern.sub('', text)
    
    def _normalize_punctuation_spacing(self, text: str) -> str:
        """
        Normalize punctuation spacing for better TTS prosody.
        Delegates to module-level function.
        """
        return _normalize_punctuation_spacing(text)


def sanitize_for_tts(text: str) -> str:
    """
    Remove URLs and emojis from text and normalize punctuation spacing for TTS.
    Use this for non-streaming responses.
    
    This function:
    - Removes URLs and emojis
    - Ensures proper spacing after punctuation (., !, ?, :, ;)
    - Normalizes whitespace (newlines, tabs, multiple spaces)
    
    Args:
        text: The text to sanitize
        
    Returns:
        Sanitized text with URLs/emojis removed and punctuation spacing normalized
    """
    if not text:
        return ""
    
    # Use a comprehensive URL pattern for non-streaming (matches full URLs)
    url_pattern = re.compile(
        r'(https?://[^\s<>"\'\)]+|www\.[^\s<>"\'\)]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s<>"\'\)]*|'
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:/[^\s<>"\'\)]*)?)'
    )
    
    # Remove URLs
    text = url_pattern.sub('', text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # Normalize punctuation spacing
    text = _normalize_punctuation_spacing(text)
    
    return text


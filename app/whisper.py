"""
Whisper Speech-to-Text Service

This module provides integration with whisper.cpp server for speech-to-text transcription.
The whisper.cpp server provides an OpenAI-compatible API for transcription.

The service is completely decoupled from the main agent and can be enabled/disabled
based on availability.
"""

import os
import httpx
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Whisper server configuration
WHISPER_SERVER_URL = os.getenv("WHISPER_SERVER_URL", "http://localhost:8080")
WHISPER_TIMEOUT = float(os.getenv("WHISPER_TIMEOUT", "60"))  # seconds


class WhisperStatus(str, Enum):
    """Status of the Whisper service"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class WhisperInfo:
    """Information about the Whisper service"""
    status: WhisperStatus
    message: str
    model: Optional[str] = None
    server_url: Optional[str] = None
    cpu_threads: Optional[int] = None


@dataclass
class TranscriptionResult:
    """Result of a transcription request"""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    language: Optional[str] = None


async def check_whisper_status() -> WhisperInfo:
    """
    Check if the Whisper server is available and get its status.
    
    Returns:
        WhisperInfo with the current status of the Whisper service
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try to get server health/status
            # whisper.cpp server doesn't have a dedicated health endpoint,
            # but we can check if the inference endpoint is responding
            response = await client.get(f"{WHISPER_SERVER_URL}/")
            
            if response.status_code == 200:
                return WhisperInfo(
                    status=WhisperStatus.AVAILABLE,
                    message="Whisper server is running and ready",
                    server_url=WHISPER_SERVER_URL,
                    model="whisper.cpp"  # We could parse this from response if available
                )
            else:
                return WhisperInfo(
                    status=WhisperStatus.ERROR,
                    message=f"Whisper server returned status {response.status_code}",
                    server_url=WHISPER_SERVER_URL
                )
                
    except httpx.ConnectError:
        return WhisperInfo(
            status=WhisperStatus.UNAVAILABLE,
            message="Cannot connect to Whisper server. Make sure whisper.cpp server is running.",
            server_url=WHISPER_SERVER_URL
        )
    except httpx.TimeoutException:
        return WhisperInfo(
            status=WhisperStatus.UNAVAILABLE,
            message="Connection to Whisper server timed out",
            server_url=WHISPER_SERVER_URL
        )
    except Exception as e:
        logger.error(f"Error checking Whisper status: {e}")
        return WhisperInfo(
            status=WhisperStatus.ERROR,
            message=f"Error checking Whisper status: {str(e)}",
            server_url=WHISPER_SERVER_URL
        )


async def transcribe_audio(
    audio_data: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = None,
    response_format: str = "json"
) -> TranscriptionResult:
    """
    Transcribe audio data using the Whisper server.
    
    Args:
        audio_data: Raw audio bytes (WAV format preferred)
        filename: Original filename for content-type detection
        language: Optional language code (e.g., 'en', 'es'). Auto-detect if None.
        response_format: Output format ('json', 'text', 'srt', 'vtt')
    
    Returns:
        TranscriptionResult with the transcribed text or error
    """
    try:
        async with httpx.AsyncClient(timeout=WHISPER_TIMEOUT) as client:
            # Prepare the multipart form data
            # whisper.cpp server uses OpenAI-compatible API
            files = {
                "file": (filename, audio_data, "audio/wav")
            }
            
            data: Dict[str, Any] = {
                "response_format": response_format,
            }
            
            if language:
                data["language"] = language
            
            # POST to /inference endpoint (whisper.cpp server)
            response = await client.post(
                f"{WHISPER_SERVER_URL}/inference",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # whisper.cpp server returns {"text": "..."}
                transcribed_text = result.get("text", "").strip()
                
                return TranscriptionResult(
                    success=True,
                    text=transcribed_text,
                    language=result.get("language"),
                    duration_ms=result.get("duration_ms")
                )
            else:
                error_text = response.text
                logger.error(f"Whisper transcription failed: {response.status_code} - {error_text}")
                return TranscriptionResult(
                    success=False,
                    error=f"Transcription failed: {error_text}"
                )
                
    except httpx.ConnectError:
        return TranscriptionResult(
            success=False,
            error="Cannot connect to Whisper server"
        )
    except httpx.TimeoutException:
        return TranscriptionResult(
            success=False,
            error="Transcription request timed out"
        )
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return TranscriptionResult(
            success=False,
            error=f"Transcription error: {str(e)}"
        )


async def test_transcription(audio_data: bytes) -> Dict[str, Any]:
    """
    Test transcription with a sample audio file.
    
    Returns a detailed result including timing information.
    """
    import time
    
    start_time = time.time()
    result = await transcribe_audio(audio_data)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        "success": result.success,
        "text": result.text,
        "error": result.error,
        "transcription_time_ms": elapsed_time,
        "audio_duration_ms": result.duration_ms,
        "language": result.language
    }


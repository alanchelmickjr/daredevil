"""Daredevil MCP server — local acoustic awareness as tools for LLMs.

Exposes the pipeline over the Model Context Protocol (stdio transport).
No audio leaves the device; the LLM receives only the structured awareness map.

Usage:
    python -m daredevil.mcp_server          # stdio (for Claude Code / Claude Desktop)
    daredevil mcp                           # same, via CLI
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import Config
from .pipeline import Pipeline
from .stage3.router import llm_payload

_server = Server("daredevil")
_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(config=Config(), warmup=True)
    return _pipeline


@_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="listen",
            description=(
                "Capture audio and return a structured acoustic awareness map: "
                "WHO is speaking (speaker ID), WHERE (spatial DOA), WHAT (event class), "
                "HOW (prosody/emotion). All inference is local — no audio leaves the device."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "Seconds of audio to analyze (default 1.0)",
                        "default": 1.0,
                    },
                    "source": {
                        "type": "string",
                        "enum": ["auto", "live", "synthetic"],
                        "description": "Audio source: live mic, synthetic demo, or auto-detect",
                        "default": "auto",
                    },
                },
                "additionalProperties": False,
            },
        ),
        Tool(
            name="enroll_speaker",
            description=(
                "Enroll a speaker's voiceprint for future identification. "
                "Records a short clip and stores a non-reversible embedding vector. "
                "No raw audio is persisted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Label for the speaker (e.g. first name)",
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Recording duration (default 3s, more = higher confidence)",
                        "default": 3.0,
                    },
                    "source": {
                        "type": "string",
                        "enum": ["auto", "live", "synthetic"],
                        "default": "auto",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="devices",
            description=(
                "Show detected mic array, installed backends (ECAPA/PANNs/librosa), "
                "enrolled speakers, and inference device (MPS/CUDA/CPU/fallback)."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        Tool(
            name="awareness",
            description=(
                "Listen and return ONLY the sources that pass the attention gate — "
                "the subset routed to the LLM. Equivalent to listen() but pre-filtered "
                "to what matters for conversation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "default": 1.0,
                    },
                    "source": {
                        "type": "string",
                        "enum": ["auto", "live", "synthetic"],
                        "default": "auto",
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]


@_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    pipe = _get_pipeline()

    if name == "listen":
        duration = arguments.get("duration", 1.0)
        source = arguments.get("source", "auto")
        amap = pipe.listen(duration=duration, source=source)
        return [TextContent(type="text", text=json.dumps(amap, indent=2))]

    if name == "enroll_speaker":
        speaker_name = arguments["name"]
        seconds = arguments.get("seconds", 3.0)
        source = arguments.get("source", "auto")
        result = pipe.enroll(speaker_name, mic_seconds=seconds, source=source)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "devices":
        return [TextContent(type="text", text=json.dumps(pipe.devices(), indent=2))]

    if name == "awareness":
        duration = arguments.get("duration", 1.0)
        source = arguments.get("source", "auto")
        amap = pipe.listen(duration=duration, source=source)
        surfaced = llm_payload(amap)
        result = {
            "surfaced_sources": surfaced,
            "count": len(surfaced),
            "timestamp": amap["timestamp"],
            "backend": amap["backend"],
            "privacy": amap["privacy"],
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(read_stream, write_stream, _server.create_initialization_options())


def main():
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()

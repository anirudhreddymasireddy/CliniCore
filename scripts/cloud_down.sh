#!/usr/bin/env bash
pkill -f "uvicorn main:app"; pkill -f "cloudflared tunnel"; echo "stopped"

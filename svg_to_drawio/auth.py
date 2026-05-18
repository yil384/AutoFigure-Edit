"""Anthropic client factory.

Picks api_key vs auth_token based on the token shape so callers don't have to
care which credential the user supplies. Returns None when no credential is
available (callers should fall back gracefully).
"""
from __future__ import annotations

import os


def make_anthropic_client():
    try:
        import anthropic
    except ImportError:
        return None
    key = os.environ.get('ANTHROPIC_API_KEY')
    oauth = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN')
    if key and not key.startswith('sk-ant-oat'):
        return anthropic.Anthropic(api_key=key)
    token = oauth or key
    if not token:
        return None
    return anthropic.Anthropic(
        auth_token=token,
        default_headers={'anthropic-beta': 'oauth-2025-04-20'},
    )

# Keepalive Domain
# Maintains Cloud Run instance activity during background tab operations

from domains.keepalive.api import keepalive_bp

__all__ = ['keepalive_bp']

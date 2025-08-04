"""
Autentifikacija i autorizacija za RowData modul
"""
import os
import jwt
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from flask import request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from ..config.settings import SECURITY_CONFIG, RATE_LIMITS
from .exceptions import AuthenticationError, AuthorizationError, RateLimitError


# Inicijalizacija rate limiter-a
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=f"redis://localhost:6379/1",  # Koristi istu Redis DB kao RowData
    storage_options={"socket_connect_timeout": 30},
    on_breach=lambda: RateLimitError("Too many requests")
)


class JWTManager:
    """Manager za JWT tokene"""
    
    def __init__(self):
        self.secret = SECURITY_CONFIG['jwt_secret']
        self.algorithm = SECURITY_CONFIG['jwt_algorithm']
        self.expiry = SECURITY_CONFIG['jwt_expiry']
    
    def generate_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """Generiše JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=self.expiry),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        if additional_claims:
            payload.update(additional_claims)
            
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verifikuje JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def refresh_token(self, token: str) -> str:
        """Osvežava token ako je valjan"""
        payload = self.verify_token(token)
        user_id = payload.get('user_id')
        
        # Generiši novi token sa istim user_id
        return self.generate_token(user_id)


# Globalna instanca JWT manager-a
jwt_manager = JWTManager()


def extract_token_from_request() -> Optional[str]:
    """Ekstraktuje token iz request-a"""
    # Proveri Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            # Format: "Bearer <token>"
            parts = auth_header.split(' ')
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                return parts[1]
        except Exception:
            pass
    
    # Proveri query parameter
    token = request.args.get('token')
    if token:
        return token
    
    # Proveri cookie
    token = request.cookies.get('rowdata_token')
    if token:
        return token
    
    return None


def require_auth(f: Callable) -> Callable:
    """Decorator koji zahteva autentifikaciju"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Proveri da li je autentifikacija uključena
        if not SECURITY_CONFIG['require_auth']:
            return f(*args, **kwargs)
        
        # Ekstraktuj token
        token = extract_token_from_request()
        if not token:
            raise AuthenticationError("No authentication token provided")
        
        # Verifikuj token
        try:
            payload = jwt_manager.verify_token(token)
            # Dodaj user info u request context
            request.rowdata_user = {
                'user_id': payload.get('user_id'),
                'token_type': payload.get('type')
            }
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {str(e)}")
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_permission(permission: str) -> Callable:
    """Decorator koji zahteva specifičnu dozvolu"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Prvo proveri autentifikaciju
            if not hasattr(request, 'rowdata_user'):
                raise AuthenticationError("Authentication required")
            
            # Ovde bi trebalo proveriti da li user ima potrebnu dozvolu
            # Za sada, samo proveravamo da li je autentifikovan
            # TODO: Implementirati permission sistem
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def apply_rate_limit(limit_string: str) -> Callable:
    """Primenjuje rate limit na endpoint"""
    def decorator(f: Callable) -> Callable:
        # Koristi Flask-Limiter decorator
        limited = limiter.limit(limit_string)(f)
        
        @wraps(limited)
        def decorated_function(*args, **kwargs):
            try:
                return limited(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    raise RateLimitError("Too many requests", retry_after=60)
                raise
        
        return decorated_function
    
    return decorator


def validate_api_key(api_key: str) -> bool:
    """Validira API ključ (alternativa JWT-u)"""
    # TODO: Implementirati API key validaciju
    # Za sada, samo provera da li postoji
    return bool(api_key and len(api_key) >= 32)


def check_cors_origin(origin: str) -> bool:
    """Proverava da li je origin dozvoljen"""
    allowed_origins = SECURITY_CONFIG['allowed_origins']
    
    if '*' in allowed_origins:
        return True
    
    return origin in allowed_origins


class PermissionChecker:
    """Klasa za proveru dozvola"""
    
    PERMISSIONS = {
        'upload': ['user', 'admin'],
        'download': ['user', 'admin'],
        'delete': ['admin'],
        'view_all': ['admin']
    }
    
    @staticmethod
    def has_permission(user_role: str, action: str) -> bool:
        """Proverava da li role ima dozvolu za akciju"""
        allowed_roles = PermissionChecker.PERMISSIONS.get(action, [])
        return user_role in allowed_roles
    
    @staticmethod
    def check_resource_ownership(user_id: str, resource_id: str) -> bool:
        """Proverava da li user poseduje resurs"""
        # TODO: Implementirati proveru vlasništva
        # Za sada, uvek vraća True
        return True


# Error handler za autentifikacione greške
def handle_auth_error(error: AuthenticationError) -> tuple:
    """Handler za autentifikacione greške"""
    return jsonify({
        "error": error.code,
        "message": error.message
    }), error.status_code


# Middleware za CORS
def setup_cors_headers(response):
    """Postavlja CORS header-e"""
    origin = request.headers.get('Origin')
    
    if origin and check_cors_origin(origin):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response
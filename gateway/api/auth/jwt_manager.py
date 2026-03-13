"""JWT token management for user authentication."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from pydantic import BaseModel


class TokenData(BaseModel):
    """Token payload data model."""
    user_id: str
    username: str
    exp: Optional[datetime] = None


class JWTManager:
    """Manager for creating and verifying JWT tokens."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        expire_minutes: int = 1440  # 24 hours
    ):
        """
        Initialize JWT manager.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
            expire_minutes: Token expiration time in minutes (default: 1440 = 24 hours)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expire_minutes = expire_minutes
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new JWT access token.
        
        Args:
            user_id: User ID to encode in token
            username: Username to encode in token
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = {
            "user_id": user_id,
            "username": username
        }
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode a JWT token without verification.
        
        Args:
            token: JWT token string to decode
            
        Returns:
            Decoded token payload as dictionary
            
        Raises:
            JWTError: If token cannot be decoded
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False}
            )
            return payload
        except JWTError as e:
            raise JWTError(f"Failed to decode token: {str(e)}")
    
    def verify_token(self, token: str) -> TokenData:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string to verify
            
        Returns:
            TokenData object with user information
            
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            user_id: str = payload.get("user_id")
            username: str = payload.get("username")
            exp: Optional[int] = payload.get("exp")
            
            if user_id is None or username is None:
                raise JWTError("Token missing required fields")
            
            exp_datetime = datetime.fromtimestamp(exp) if exp else None
            
            return TokenData(
                user_id=user_id,
                username=username,
                exp=exp_datetime
            )
        except JWTError as e:
            raise JWTError(f"Token verification failed: {str(e)}")
    
    def is_token_expired(self, token: str) -> bool:
        """
        Check if a token is expired.
        
        Args:
            token: JWT token string to check
            
        Returns:
            True if token is expired, False otherwise
        """
        try:
            payload = self.decode_token(token)
            exp = payload.get("exp")
            
            if exp is None:
                return True
            
            exp_datetime = datetime.fromtimestamp(exp)
            return datetime.utcnow() > exp_datetime
        except JWTError:
            return True
    
    def refresh_token(self, token: str) -> str:
        """
        Refresh an existing token by creating a new one with the same user data.
        
        Args:
            token: Existing JWT token to refresh
            
        Returns:
            New JWT token string
            
        Raises:
            JWTError: If token is invalid
        """
        try:
            token_data = self.verify_token(token)
            
            # Create new token with same user data
            new_token = self.create_access_token(
                user_id=token_data.user_id,
                username=token_data.username
            )
            
            return new_token
        except JWTError as e:
            raise JWTError(f"Failed to refresh token: {str(e)}")

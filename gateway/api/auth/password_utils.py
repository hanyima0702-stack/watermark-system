"""Password hashing and verification utilities using bcrypt."""

import re
from typing import Tuple
from passlib.context import CryptContext


# Configure password context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string
        
    Raises:
        ValueError: If password is empty or None
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    if not plain_password or not hashed_password:
        return False
    
    return pwd_context.verify(plain_password, hashed_password)


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password strength according to security requirements.
    
    Requirements:
    - At least 8 characters long
    - Contains at least one letter
    - Contains at least one number
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not password:
        return False, "密码不能为空"
    
    if len(password) < 8:
        return False, "密码长度至少为8位"
    
    # Check for at least one letter
    if not re.search(r'[a-zA-Z]', password):
        return False, "密码必须包含至少一个字母"
    
    # Check for at least one number
    if not re.search(r'\d', password):
        return False, "密码必须包含至少一个数字"
    
    return True, ""


def is_password_strong(password: str) -> bool:
    """
    Check if password meets strength requirements.
    
    Args:
        password: Password to check
        
    Returns:
        True if password is strong enough, False otherwise
    """
    is_valid, _ = validate_password_strength(password)
    return is_valid

"""Unit tests for password utilities."""

import pytest
from gateway.api.auth.password_utils import (
    hash_password,
    verify_password,
    validate_password_strength,
    is_password_strong
)


class TestHashPassword:
    """Tests for hash_password function."""
    
    def test_hash_password_success(self):
        """Test successful password hashing."""
        password = "TestPassword123"
        hashed = hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt prefix
    
    def test_hash_password_different_hashes(self):
        """Test that same password produces different hashes (due to salt)."""
        password = "TestPassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2
    
    def test_hash_password_empty_raises_error(self):
        """Test that empty password raises ValueError."""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            hash_password("")
    
    def test_hash_password_none_raises_error(self):
        """Test that None password raises ValueError."""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            hash_password(None)


class TestVerifyPassword:
    """Tests for verify_password function."""
    
    def test_verify_password_correct(self):
        """Test verification with correct password."""
        password = "TestPassword123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test verification with incorrect password."""
        password = "TestPassword123"
        wrong_password = "WrongPassword456"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_verify_password_empty_plain(self):
        """Test verification with empty plain password."""
        hashed = hash_password("TestPassword123")
        
        assert verify_password("", hashed) is False
    
    def test_verify_password_empty_hash(self):
        """Test verification with empty hash."""
        assert verify_password("TestPassword123", "") is False
    
    def test_verify_password_none_values(self):
        """Test verification with None values."""
        assert verify_password(None, "somehash") is False
        assert verify_password("password", None) is False


class TestValidatePasswordStrength:
    """Tests for validate_password_strength function."""
    
    def test_valid_password(self):
        """Test validation of valid password."""
        is_valid, message = validate_password_strength("Password123")
        
        assert is_valid is True
        assert message == ""
    
    def test_password_too_short(self):
        """Test validation of password that is too short."""
        is_valid, message = validate_password_strength("Pass1")
        
        assert is_valid is False
        assert "至少为8位" in message
    
    def test_password_no_letter(self):
        """Test validation of password without letters."""
        is_valid, message = validate_password_strength("12345678")
        
        assert is_valid is False
        assert "至少一个字母" in message
    
    def test_password_no_number(self):
        """Test validation of password without numbers."""
        is_valid, message = validate_password_strength("PasswordOnly")
        
        assert is_valid is False
        assert "至少一个数字" in message
    
    def test_password_empty(self):
        """Test validation of empty password."""
        is_valid, message = validate_password_strength("")
        
        assert is_valid is False
        assert "不能为空" in message
    
    def test_password_none(self):
        """Test validation of None password."""
        is_valid, message = validate_password_strength(None)
        
        assert is_valid is False
        assert "不能为空" in message
    
    def test_password_exactly_8_chars(self):
        """Test validation of password with exactly 8 characters."""
        is_valid, message = validate_password_strength("Pass1234")
        
        assert is_valid is True
        assert message == ""
    
    def test_password_with_special_chars(self):
        """Test validation of password with special characters."""
        is_valid, message = validate_password_strength("Pass@123!")
        
        assert is_valid is True
        assert message == ""


class TestIsPasswordStrong:
    """Tests for is_password_strong function."""
    
    def test_strong_password(self):
        """Test strong password returns True."""
        assert is_password_strong("Password123") is True
    
    def test_weak_password(self):
        """Test weak password returns False."""
        assert is_password_strong("weak") is False
    
    def test_password_no_number(self):
        """Test password without number returns False."""
        assert is_password_strong("PasswordOnly") is False
    
    def test_password_no_letter(self):
        """Test password without letter returns False."""
        assert is_password_strong("12345678") is False

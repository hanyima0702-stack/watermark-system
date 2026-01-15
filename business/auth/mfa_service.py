"""
多因素认证服务
提供TOTP、SMS、邮件等多种二次认证方式
"""

import pyotp
import qrcode
import secrets
import string
from io import BytesIO
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import smtplib
import base64

from ...shared.config import get_settings
from .models import MFAMethod, MFAChallenge, MFASetupRequest


class MFAService:
    """多因素认证服务"""
    
    def __init__(self):
        self.settings = get_settings()
        self.app_name = self.settings.app_name
        self.challenges: Dict[str, MFAChallenge] = {}  # 实际应该存储在Redis中
    
    def setup_totp(self, user_id: str, username: str) -> MFAChallenge:
        """设置TOTP认证"""
        # 生成密钥
        secret = pyotp.random_base32()
        
        # 创建TOTP对象
        totp = pyotp.TOTP(secret)
        
        # 生成二维码
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name=self.app_name
        )
        
        qr_code = self._generate_qr_code(provisioning_uri)
        
        # 生成备用码
        backup_codes = self._generate_backup_codes()
        
        # 创建挑战
        challenge_id = self._generate_challenge_id()
        challenge = MFAChallenge(
            challenge_id=challenge_id,
            method=MFAMethod.TOTP,
            expires_at=datetime.utcnow() + timedelta(minutes=10),
            qr_code=qr_code,
            backup_codes=backup_codes
        )
        
        # 存储挑战（实际应该存储在Redis中）
        self.challenges[challenge_id] = challenge
        
        # 存储用户的TOTP密钥（实际应该加密存储在数据库中）
        self._store_user_totp_secret(user_id, secret)
        
        return challenge
    
    def setup_sms(self, user_id: str, phone_number: str) -> MFAChallenge:
        """设置SMS认证"""
        # 生成验证码
        verification_code = self._generate_verification_code()
        
        # 发送SMS（这里需要集成SMS服务提供商）
        success = self._send_sms(phone_number, verification_code)
        
        if not success:
            raise Exception("SMS发送失败")
        
        # 创建挑战
        challenge_id = self._generate_challenge_id()
        challenge = MFAChallenge(
            challenge_id=challenge_id,
            method=MFAMethod.SMS,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        # 存储挑战和验证码
        self.challenges[challenge_id] = challenge
        self._store_verification_code(challenge_id, verification_code)
        
        # 存储用户手机号
        self._store_user_phone(user_id, phone_number)
        
        return challenge
    
    def setup_email(self, user_id: str, email: str) -> MFAChallenge:
        """设置邮件认证"""
        # 生成验证码
        verification_code = self._generate_verification_code()
        
        # 发送邮件
        success = self._send_email(email, verification_code)
        
        if not success:
            raise Exception("邮件发送失败")
        
        # 创建挑战
        challenge_id = self._generate_challenge_id()
        challenge = MFAChallenge(
            challenge_id=challenge_id,
            method=MFAMethod.EMAIL,
            expires_at=datetime.utcnow() + timedelta(minutes=10)
        )
        
        # 存储挑战和验证码
        self.challenges[challenge_id] = challenge
        self._store_verification_code(challenge_id, verification_code)
        
        return challenge
    
    def verify_totp(self, user_id: str, code: str) -> bool:
        """验证TOTP代码"""
        secret = self._get_user_totp_secret(user_id)
        if not secret:
            return False
        
        totp = pyotp.TOTP(secret)
        
        # 验证当前代码
        if totp.verify(code):
            return True
        
        # 检查是否为备用码
        return self._verify_backup_code(user_id, code)
    
    def verify_sms_code(self, challenge_id: str, code: str) -> bool:
        """验证SMS代码"""
        challenge = self.challenges.get(challenge_id)
        if not challenge or challenge.is_expired():
            return False
        
        stored_code = self._get_verification_code(challenge_id)
        return stored_code == code
    
    def verify_email_code(self, challenge_id: str, code: str) -> bool:
        """验证邮件代码"""
        return self.verify_sms_code(challenge_id, code)  # 逻辑相同
    
    def generate_mfa_challenge(self, user_id: str, method: MFAMethod) -> MFAChallenge:
        """生成MFA挑战"""
        if method == MFAMethod.TOTP:
            # TOTP不需要发送挑战，直接返回空挑战
            challenge_id = self._generate_challenge_id()
            return MFAChallenge(
                challenge_id=challenge_id,
                method=method,
                expires_at=datetime.utcnow() + timedelta(minutes=5)
            )
        
        elif method == MFAMethod.SMS:
            phone = self._get_user_phone(user_id)
            if not phone:
                raise Exception("用户未设置手机号")
            return self.setup_sms(user_id, phone)
        
        elif method == MFAMethod.EMAIL:
            email = self._get_user_email(user_id)
            if not email:
                raise Exception("用户未设置邮箱")
            return self.setup_email(user_id, email)
        
        else:
            raise Exception(f"不支持的MFA方法: {method}")
    
    def verify_mfa_code(self, user_id: str, challenge_id: str, code: str) -> bool:
        """验证MFA代码"""
        challenge = self.challenges.get(challenge_id)
        if not challenge or challenge.is_expired():
            return False
        
        if challenge.method == MFAMethod.TOTP:
            return self.verify_totp(user_id, code)
        elif challenge.method == MFAMethod.SMS:
            return self.verify_sms_code(challenge_id, code)
        elif challenge.method == MFAMethod.EMAIL:
            return self.verify_email_code(challenge_id, code)
        
        return False
    
    def is_mfa_enabled(self, user_id: str) -> bool:
        """检查用户是否启用了MFA"""
        # 检查用户是否设置了任何MFA方法
        return (self._get_user_totp_secret(user_id) is not None or
                self._get_user_phone(user_id) is not None)
    
    def get_user_mfa_methods(self, user_id: str) -> List[MFAMethod]:
        """获取用户启用的MFA方法"""
        methods = []
        
        if self._get_user_totp_secret(user_id):
            methods.append(MFAMethod.TOTP)
        
        if self._get_user_phone(user_id):
            methods.append(MFAMethod.SMS)
        
        # 邮件MFA通常基于用户邮箱，总是可用
        if self._get_user_email(user_id):
            methods.append(MFAMethod.EMAIL)
        
        return methods
    
    def disable_mfa(self, user_id: str, method: MFAMethod) -> bool:
        """禁用指定的MFA方法"""
        try:
            if method == MFAMethod.TOTP:
                self._remove_user_totp_secret(user_id)
            elif method == MFAMethod.SMS:
                self._remove_user_phone(user_id)
            # 邮件MFA不需要特殊处理，因为基于用户邮箱
            
            return True
        except Exception:
            return False
    
    def _generate_qr_code(self, data: str) -> str:
        """生成二维码"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # 转换为base64字符串
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """生成备用码"""
        codes = []
        for _ in range(count):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes
    
    def _generate_verification_code(self, length: int = 6) -> str:
        """生成验证码"""
        return ''.join(secrets.choice(string.digits) for _ in range(length))
    
    def _generate_challenge_id(self) -> str:
        """生成挑战ID"""
        return secrets.token_urlsafe(32)
    
    def _send_sms(self, phone_number: str, code: str) -> bool:
        """发送SMS（需要集成SMS服务提供商）"""
        # 这里应该集成实际的SMS服务提供商，如阿里云、腾讯云等
        print(f"发送SMS到 {phone_number}: 验证码 {code}")
        return True  # 模拟发送成功
    
    def _send_email(self, email: str, code: str) -> bool:
        """发送邮件"""
        if not self.settings.email.enabled:
            print(f"邮件服务未启用，验证码: {code}")
            return True  # 开发环境模拟
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.settings.email.from_email
            msg['To'] = email
            msg['Subject'] = f"{self.app_name} - 验证码"
            
            body = f"""
            您的验证码是: {code}
            
            此验证码将在10分钟后过期。
            如果您没有请求此验证码，请忽略此邮件。
            
            {self.app_name}
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.settings.email.smtp_host, self.settings.email.smtp_port)
            if self.settings.email.smtp_tls:
                server.starttls()
            
            if self.settings.email.smtp_user and self.settings.email.smtp_password:
                server.login(self.settings.email.smtp_user, self.settings.email.smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"邮件发送失败: {e}")
            return False
    
    # 以下方法在实际实现中应该与数据库和缓存交互
    def _store_user_totp_secret(self, user_id: str, secret: str):
        """存储用户TOTP密钥"""
        # 实际应该加密存储在数据库中
        pass
    
    def _get_user_totp_secret(self, user_id: str) -> Optional[str]:
        """获取用户TOTP密钥"""
        # 实际应该从数据库中获取并解密
        return None
    
    def _remove_user_totp_secret(self, user_id: str):
        """删除用户TOTP密钥"""
        pass
    
    def _store_user_phone(self, user_id: str, phone: str):
        """存储用户手机号"""
        pass
    
    def _get_user_phone(self, user_id: str) -> Optional[str]:
        """获取用户手机号"""
        return None
    
    def _remove_user_phone(self, user_id: str):
        """删除用户手机号"""
        pass
    
    def _get_user_email(self, user_id: str) -> Optional[str]:
        """获取用户邮箱"""
        return None
    
    def _store_verification_code(self, challenge_id: str, code: str):
        """存储验证码"""
        # 实际应该存储在Redis中，设置过期时间
        pass
    
    def _get_verification_code(self, challenge_id: str) -> Optional[str]:
        """获取验证码"""
        return None
    
    def _verify_backup_code(self, user_id: str, code: str) -> bool:
        """验证备用码"""
        # 实际应该从数据库中获取用户的备用码列表进行验证
        # 验证成功后应该将该备用码标记为已使用
        return False
"""初始数据库迁移

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 创建扩展
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    
    # 用户表
    op.create_table('users',
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('department', sa.String(length=100), nullable=True),
        sa.Column('roles', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('password_hash', sa.String(length=255), nullable=True),
        sa.Column('salt', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    op.create_index(op.f('ix_users_is_active'), 'users', ['is_active'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)
    
    # 文件元数据表
    op.create_table('file_metadata',
        sa.Column('file_id', sa.String(length=50), nullable=False),
        sa.Column('original_name', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=100), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('uploaded_by', sa.String(length=50), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['uploaded_by'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('file_id')
    )
    op.create_index('idx_file_hash', 'file_metadata', ['file_hash'], unique=False)
    op.create_index('idx_file_hash_unique', 'file_metadata', ['file_hash', 'uploaded_by'], unique=False)
    op.create_index('idx_file_metadata_composite', 'file_metadata', ['uploaded_by', 'file_type', 'uploaded_at'], unique=False)
    op.create_index(op.f('ix_file_metadata_file_type'), 'file_metadata', ['file_type'], unique=False)
    op.create_index(op.f('ix_file_metadata_uploaded_at'), 'file_metadata', ['uploaded_at'], unique=False)
    op.create_index(op.f('ix_file_metadata_uploaded_by'), 'file_metadata', ['uploaded_by'], unique=False)
    
    # 水印配置表
    op.create_table('watermark_configs',
        sa.Column('config_id', sa.String(length=50), nullable=False),
        sa.Column('config_name', sa.String(length=100), nullable=False),
        sa.Column('watermark_type', sa.String(length=20), nullable=False),
        sa.Column('visible_config', sa.JSON(), nullable=True),
        sa.Column('invisible_config', sa.JSON(), nullable=True),
        sa.Column('template_variables', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_by', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("watermark_type IN ('visible', 'invisible', 'both')", name='check_watermark_type'),
        sa.ForeignKeyConstraint(['created_by'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('config_id')
    )
    op.create_index('idx_config_name_user', 'watermark_configs', ['config_name', 'created_by'], unique=False)
    op.create_index(op.f('ix_watermark_configs_config_name'), 'watermark_configs', ['config_name'], unique=False)
    op.create_index(op.f('ix_watermark_configs_created_by'), 'watermark_configs', ['created_by'], unique=False)
    op.create_index(op.f('ix_watermark_configs_is_active'), 'watermark_configs', ['is_active'], unique=False)
    op.create_index(op.f('ix_watermark_configs_watermark_type'), 'watermark_configs', ['watermark_type'], unique=False)
    
    # 水印任务表
    op.create_table('watermark_tasks',
        sa.Column('task_id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('file_id', sa.String(length=50), nullable=False),
        sa.Column('config_id', sa.String(length=50), nullable=False),
        sa.Column('task_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('progress', sa.DECIMAL(precision=5, scale=2), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('processing_engine', sa.String(length=50), nullable=True),
        sa.Column('output_file_id', sa.String(length=50), nullable=True),
        sa.Column('processing_time', sa.DECIMAL(precision=10, scale=3), nullable=True),
        sa.Column('quality_metrics', sa.JSON(), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("progress >= 0.00 AND progress <= 100.00", name='check_progress_range'),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", name='check_task_status'),
        sa.ForeignKeyConstraint(['config_id'], ['watermark_configs.config_id'], ),
        sa.ForeignKeyConstraint(['file_id'], ['file_metadata.file_id'], ),
        sa.ForeignKeyConstraint(['output_file_id'], ['file_metadata.file_id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('task_id')
    )
    op.create_index('idx_task_type_status', 'watermark_tasks', ['task_type', 'status'], unique=False)
    op.create_index('idx_watermark_tasks_composite', 'watermark_tasks', ['user_id', 'status', 'created_at'], unique=False)
    op.create_index(op.f('ix_watermark_tasks_file_id'), 'watermark_tasks', ['file_id'], unique=False)
    op.create_index(op.f('ix_watermark_tasks_status'), 'watermark_tasks', ['status'], unique=False)
    op.create_index(op.f('ix_watermark_tasks_task_type'), 'watermark_tasks', ['task_type'], unique=False)
    op.create_index(op.f('ix_watermark_tasks_user_id'), 'watermark_tasks', ['user_id'], unique=False)
    
    # 水印提取结果表
    op.create_table('extraction_results',
        sa.Column('result_id', sa.String(length=50), nullable=False),
        sa.Column('file_id', sa.String(length=50), nullable=False),
        sa.Column('extracted_user_id', sa.String(length=50), nullable=True),
        sa.Column('extracted_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('confidence_score', sa.DECIMAL(precision=5, scale=4), nullable=True),
        sa.Column('extraction_method', sa.String(length=50), nullable=False),
        sa.Column('extraction_details', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['extracted_user_id'], ['users.user_id'], ),
        sa.ForeignKeyConstraint(['file_id'], ['file_metadata.file_id'], ),
        sa.PrimaryKeyConstraint('result_id')
    )
    op.create_index('idx_extraction_method_confidence', 'extraction_results', ['extraction_method', 'confidence_score'], unique=False)
    op.create_index('idx_extraction_results_composite', 'extraction_results', ['file_id', 'confidence_score', 'created_at'], unique=False)
    op.create_index(op.f('ix_extraction_results_confidence_score'), 'extraction_results', ['confidence_score'], unique=False)
    op.create_index(op.f('ix_extraction_results_created_at'), 'extraction_results', ['created_at'], unique=False)
    op.create_index(op.f('ix_extraction_results_extraction_method'), 'extraction_results', ['extraction_method'], unique=False)
    op.create_index(op.f('ix_extraction_results_extracted_user_id'), 'extraction_results', ['extracted_user_id'], unique=False)
    op.create_index(op.f('ix_extraction_results_file_id'), 'extraction_results', ['file_id'], unique=False)
    
    # 证据报告表
    op.create_table('evidence_reports',
        sa.Column('report_id', sa.String(length=50), nullable=False),
        sa.Column('result_id', sa.String(length=50), nullable=False),
        sa.Column('report_path', sa.String(length=500), nullable=False),
        sa.Column('report_metadata', sa.JSON(), nullable=True),
        sa.Column('generated_by', sa.String(length=50), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['generated_by'], ['users.user_id'], ),
        sa.ForeignKeyConstraint(['result_id'], ['extraction_results.result_id'], ),
        sa.PrimaryKeyConstraint('report_id')
    )
    op.create_index('idx_evidence_reports_composite', 'evidence_reports', ['result_id', 'generated_by', 'generated_at'], unique=False)
    op.create_index(op.f('ix_evidence_reports_generated_at'), 'evidence_reports', ['generated_at'], unique=False)
    op.create_index(op.f('ix_evidence_reports_generated_by'), 'evidence_reports', ['generated_by'], unique=False)
    op.create_index(op.f('ix_evidence_reports_result_id'), 'evidence_reports', ['result_id'], unique=False)
    
    # 审计日志表
    op.create_table('audit_logs',
        sa.Column('log_id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.String(length=50), nullable=True),
        sa.Column('ip_address', postgresql.INET(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('log_id')
    )
    op.create_index('idx_audit_logs_composite', 'audit_logs', ['user_id', 'action', 'timestamp'], unique=False)
    op.create_index('idx_audit_logs_resource', 'audit_logs', ['resource_type', 'resource_id', 'timestamp'], unique=False)
    op.create_index('idx_audit_logs_time_range', 'audit_logs', ['timestamp', 'action'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_id'), 'audit_logs', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_type'), 'audit_logs', ['resource_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    
    # 系统配置表
    op.create_table('system_configs',
        sa.Column('config_key', sa.String(length=100), nullable=False),
        sa.Column('config_value', sa.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_encrypted', sa.Boolean(), nullable=True),
        sa.Column('updated_by', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['updated_by'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('config_key')
    )
    
    # 密钥管理表
    op.create_table('key_management',
        sa.Column('key_id', sa.String(length=50), nullable=False),
        sa.Column('key_type', sa.String(length=50), nullable=False),
        sa.Column('key_data', sa.Text(), nullable=False),
        sa.Column('key_version', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.String(length=50), nullable=False),
        sa.Column('rotated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("key_type IN ('encryption', 'signing', 'watermark')", name='check_key_type'),
        sa.ForeignKeyConstraint(['created_by'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('key_id')
    )
    op.create_index('idx_key_management_composite', 'key_management', ['key_type', 'is_active', 'expires_at'], unique=False)
    op.create_index(op.f('ix_key_management_expires_at'), 'key_management', ['expires_at'], unique=False)
    op.create_index(op.f('ix_key_management_is_active'), 'key_management', ['is_active'], unique=False)
    op.create_index(op.f('ix_key_management_key_type'), 'key_management', ['key_type'], unique=False)
    
    # 创建更新时间触发器函数
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # 为需要的表添加更新时间触发器
    op.execute("""
        CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_watermark_configs_updated_at BEFORE UPDATE ON watermark_configs
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_watermark_tasks_updated_at BEFORE UPDATE ON watermark_tasks
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_system_configs_updated_at BEFORE UPDATE ON system_configs
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_key_management_updated_at BEFORE UPDATE ON key_management
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    # 删除触发器
    op.execute("DROP TRIGGER IF EXISTS update_key_management_updated_at ON key_management;")
    op.execute("DROP TRIGGER IF EXISTS update_system_configs_updated_at ON system_configs;")
    op.execute("DROP TRIGGER IF EXISTS update_watermark_tasks_updated_at ON watermark_tasks;")
    op.execute("DROP TRIGGER IF EXISTS update_watermark_configs_updated_at ON watermark_configs;")
    op.execute("DROP TRIGGER IF EXISTS update_users_updated_at ON users;")
    
    # 删除触发器函数
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
    
    # 删除表
    op.drop_table('key_management')
    op.drop_table('system_configs')
    op.drop_table('audit_logs')
    op.drop_table('evidence_reports')
    op.drop_table('extraction_results')
    op.drop_table('watermark_tasks')
    op.drop_table('watermark_configs')
    op.drop_table('file_metadata')
    op.drop_table('users')
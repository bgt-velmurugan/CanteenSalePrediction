from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('sale', sa.Column('is_special_event', sa.Boolean(), nullable=True))
    op.execute('UPDATE sale SET is_special_event = False')
    op.alter_column('sale', 'is_special_event', nullable=False)

def downgrade():
    op.drop_column('sale', 'is_special_event')


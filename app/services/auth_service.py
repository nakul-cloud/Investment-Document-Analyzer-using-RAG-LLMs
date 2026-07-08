from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.exceptions import AuthError, ValidationError
from app.core.logging import logger


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalars().first()


async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalars().first()


async def register_user(db: AsyncSession, user_in: UserCreate) -> User:
    # Check if user already exists
    existing_user = await get_user_by_email(db, user_in.email)
    if existing_user:
        logger.warning(f"Registration failed: User {user_in.email} already exists")
        raise ValidationError(message="Email already registered")

    hashed_pw = get_password_hash(user_in.password)
    new_user = User(email=user_in.email, hashed_password=hashed_pw)
    db.add(new_user)
    await db.flush()  # Populates new_user.id
    
    logger.info(f"User {new_user.email} registered successfully with ID {new_user.id}")
    return new_user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> str:
    user = await get_user_by_email(db, email)
    if not user:
        logger.warning(f"Authentication failed: User {email} not found")
        raise AuthError(message="Incorrect email or password")

    if not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed: Incorrect password for user {email}")
        raise AuthError(message="Incorrect email or password")

    if not user.is_active:
        logger.warning(f"Authentication failed: User {email} is inactive")
        raise AuthError(message="Inactive user account")

    # Generate JWT token
    token = create_access_token(subject=user.id)
    logger.info(f"User {email} authenticated successfully. JWT token generated.")
    return token

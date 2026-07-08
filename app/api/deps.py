from typing import AsyncGenerator
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AuthError
from app.core.logging import logger
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import TokenPayload
from app.services import auth_service

# OAuth2 schema for Swagger UI integration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """Decodes JWT bearer tokens and resolves active User records from the database."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=["HS256"]
        )
        token_data = TokenPayload(**payload)
        if token_data.sub is None:
            raise AuthError(message="Invalid token payload: missing subject")
    except (jwt.PyJWTError, ValidationError) as e:
        logger.warning(f"JWT Token validation failed: {e}")
        raise AuthError(message="Could not validate credentials")

    user = await auth_service.get_user_by_id(db, token_data.sub)
    if not user:
        logger.warning(f"Credentials validation failed: User ID {token_data.sub} not found")
        raise AuthError(message="User not found")
        
    if not user.is_active:
        logger.warning(f"Credentials validation failed: User ID {token_data.sub} is disabled")
        raise AuthError(message="User account is inactive")
        
    return user

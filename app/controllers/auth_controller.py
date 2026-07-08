from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserCreate, UserResponse
from app.schemas.auth import Token
from app.services import auth_service
from app.core.exceptions import AppException, ValidationError
from app.core.logging import logger


class AuthController:
    @staticmethod
    async def register(db: AsyncSession, user_in: UserCreate) -> UserResponse:
        """Controller orchestrating registration checks and exceptions."""
        try:
            user = await auth_service.register_user(db, user_in)
            return UserResponse.model_validate(user)
        except AppException as e:
            # Re-raise known application exceptions
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during user registration: {e}")
            raise ValidationError(message="Failed to register user account. Please check details.")

    @staticmethod
    async def login(db: AsyncSession, email: str, password: str) -> Token:
        """Controller orchestrating authentication validation and exceptions."""
        try:
            token = await auth_service.authenticate_user(db, email, password)
            return Token(access_token=token, token_type="bearer")
        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during user login: {e}")
            raise ValidationError(message="Authentication process encountered an unexpected error.")

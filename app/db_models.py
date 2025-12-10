from datetime import datetime
from typing import Optional, List, Any
from bson import ObjectId
from pydantic import BaseModel, Field, HttpUrl, field_validator, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pymongo import IndexModel

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        from pydantic_core import core_schema
        # Use any_schema to accept both string and ObjectId inputs
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x) if x else None
            ),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if not ObjectId.is_valid(v):
                raise ValueError("Invalid objectid")
            return ObjectId(v)
        raise ValueError(f"Invalid objectid: expected str or ObjectId, got {type(v)}")

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema: JsonSchemaValue, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema.update(type="string")
        return field_schema

class User(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password_hash: str  # Will store hashed password
    role: str = Field(default="user", description="User role: admin, event_organiser, or user")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Event(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    title: str
    description: Optional[str] = None
    is_online: bool
    location_address: Optional[str] = None
    online_url: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    cost: float = 0.0
    tags: List[str] = Field(default_factory=list)
    media_urls: List[str] = Field(default_factory=list)
    created_by: PyObjectId  # User ID who created the event
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @field_validator('location_address')
    @classmethod
    def validate_location(cls, v, info):
        # If is_online is False, location_address is required
        if info.data.get('is_online') is False and not v:
            raise ValueError('location_address is required for in-person events')
        return v

# MongoDB collection indexes
event_indexes = [
    IndexModel([("start_time", 1)]),  # For chronological sorting
    IndexModel([("created_by", 1)]),  # For user's events
    IndexModel([("tags", 1)]),  # For tag filtering
]

user_indexes = [
    IndexModel([("username", 1)], unique=True),
    IndexModel([("email", 1)], unique=True),
]


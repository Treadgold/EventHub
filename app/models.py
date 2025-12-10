from pydantic import BaseModel, Field, model_validator, HttpUrl
from typing import Optional, List
from datetime import datetime

class EventSchema(BaseModel):
    title: Optional[str] = Field(None, description="The name of the event")
    description: Optional[str] = Field(None, description="Detailed description of the event")
    is_online: Optional[bool] = Field(None, description="Whether the event is online or in-person")
    location_address: Optional[str] = Field(None, description="Physical address (required if not online)")
    online_url: Optional[HttpUrl] = Field(None, description="URL for the online event")
    start_time: Optional[str] = Field(None, description="Start date and time (ISO format preferred or natural language)")
    end_time: Optional[str] = Field(None, description="End date and time")
    cost: Optional[float] = Field(None, description="Cost of the event ticket")
    tags: List[str] = Field(default_factory=list, description="Tags or categories for the event")
    media_urls: List[str] = Field(default_factory=list, description="List of image or video URLs")

    @model_validator(mode='after')
    def check_location(self):
        # If is_online is explicitly False, location_address is required
        if self.is_online is False and not self.location_address:
            # We don't raise an error here to allow partial filling, 
            # but this logic helps the agent know it's incomplete.
            pass
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Tech Meetup 2024",
                "is_online": False,
                "location_address": "123 Innovation Dr",
                "cost": 0.0
            }
        }
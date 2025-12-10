from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import os

# MongoDB connection settings
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "event_website")

# Global database client
client: Optional[AsyncIOMotorClient] = None
database = None

async def connect_to_mongo():
    """Create database connection with retry logic"""
    global client, database
    import asyncio
    
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
            database = client[DATABASE_NAME]
            # Test connection
            await client.admin.command('ping')
            print(f"Connected to MongoDB: {DATABASE_NAME}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"MongoDB connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"Failed to connect to MongoDB after {max_retries} attempts: {e}")
                raise

async def close_mongo_connection():
    """Close database connection"""
    global client
    if client:
        client.close()

def get_database():
    """Get database instance"""
    return database


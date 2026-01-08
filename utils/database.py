import asyncpg
from typing import Optional, List, Dict, Any
from config import DATABASE_URL

_db_pool: Optional[asyncpg.Pool] = None

async def init_db():
    """
    Initializes the database connection pool and creates the 'events' table if it doesn't exist.
    Also handles migration for the 'camera_id' column if it's not of type INTEGER.
    """
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0)
    
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER NOT NULL,
                object_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        try:
            await conn.execute("ALTER TABLE events ALTER COLUMN camera_id TYPE INTEGER USING camera_id::integer;")
            print("Database migration: camera_id column altered to INTEGER.")
        except Exception:
            pass

        # Create device_tokens table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS device_tokens (
                token TEXT PRIMARY KEY,
                user_id UUID,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        
        # Migrate user_id column (for existing tables with INTEGER, change to UUID)
        try:
            await conn.execute("ALTER TABLE device_tokens ADD COLUMN user_id UUID")
            print("✅ Database migration: Added user_id column to device_tokens table")
        except Exception:
            # Column already exists, try to alter type
            try:
                # Drop foreign key first if exists
                await conn.execute("ALTER TABLE device_tokens DROP CONSTRAINT IF EXISTS device_tokens_user_id_fkey")
                # Alter column type
                await conn.execute("ALTER TABLE device_tokens ALTER COLUMN user_id TYPE UUID USING user_id::text::uuid")
                print("✅ Database migration: Changed user_id column type to UUID")
            except Exception as e:
                print(f"⚠️ Migration warning: {e}")
        
        # Add foreign key constraint to auth.users if not exists
        try:
            await conn.execute("""
                ALTER TABLE device_tokens 
                ADD CONSTRAINT device_tokens_user_id_fkey 
                FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE
            """)
            print("✅ Database migration: Added foreign key constraint to auth.users")
        except Exception as e:
            print(f"⚠️ Foreign key constraint not added: {e}")

async def close_db():
    """
    Closes the database connection pool gracefully.
    """
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None

async def insert_event(camera_id: int, object_name: str, event_type: str):
    """
    Inserts a new event into the database.

    Args:
        camera_id (int): The ID of the camera that triggered the event.
        object_name (str): The name/path of the object (e.g., S3 object key).
        event_type (str): The type of event (e.g., 'frame' or 'video').
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO events(camera_id, object_name, event_type)
            VALUES ($1, $2, $3)
            """,
            camera_id,
            object_name,
            event_type,
        )

async def list_events(camera_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves a list of events for a specific camera, ordered by creation time descending.

    Args:
        camera_id (int): The ID of the camera to list events for.

    Returns:
        List[Dict[str, Any]]: A list of event records as dictionaries.
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT object_name, event_type, created_at
            FROM events
            WHERE camera_id = $1
            ORDER BY created_at DESC
            """,
            camera_id,
        )
        return [dict(r) for r in rows]

async def register_device_token(token: str, user_id: str):
    """
    Registers a device token for a specific user.
    
    Args:
        token (str): The device registration token from Firebase.
        user_id (str): The UUID of the user who owns this device (from auth.users.id).
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO device_tokens(token, user_id) 
               VALUES ($1, $2::uuid) 
               ON CONFLICT (token) DO UPDATE SET user_id = $2::uuid""",
            token, user_id
        )

async def delete_device_token(token: str):
    """
    Deletes a device token (e.g., when user logs out).
    
    Args:
        token (str): The device registration token to delete.
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        await conn.execute("DELETE FROM device_tokens WHERE token = $1", token)

async def get_tokens_for_camera_owner(camera_id: int) -> List[str]:
    """
    Get all device tokens belonging to the owner of the specified camera.
    
    Args:
        camera_id (int): The camera ID to get owner's tokens for.
        
    Returns:
        List[str]: List of device tokens for the camera owner.
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT dt.token 
            FROM device_tokens dt
            JOIN cameras c ON c.owner_id = dt.user_id
            WHERE c.camera_id = $1 AND dt.user_id IS NOT NULL
            """,
            camera_id
        )
        return [r["token"] for r in rows]

async def get_camera_label(camera_id: int) -> str:
    """
    Retrieves the camera label (name) by camera_id.
    Returns the label or a default string if not found.
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT label FROM cameras WHERE camera_id = $1", camera_id)
        if row:
            return row["label"]
        return f"Camera {camera_id}"

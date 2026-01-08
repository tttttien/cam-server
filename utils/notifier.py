import firebase_admin
from firebase_admin import credentials, messaging
import os
import asyncio
from typing import List, Optional

from utils.database import get_camera_label

# Path to your Firebase service account JSON file
FIREBASE_KEY_PATH = "firebase-service-account.json"

_firebase_app = None

def init_firebase():
    """
    Initializes the Firebase Admin SDK if the service account key exists.
    """
    global _firebase_app
    if _firebase_app:
        return _firebase_app

    if os.path.exists(FIREBASE_KEY_PATH):
        try:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            _firebase_app = firebase_admin.initialize_app(cred)
            print("🔥 Firebase Admin SDK initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
    else:
        print(f"⚠️ {FIREBASE_KEY_PATH} not found. Push notifications will be disabled.")
    
    return _firebase_app

async def send_fire_notification(tokens: List[str], camera_id: int):
    """
    Sends a push notification to multiple devices when fire is detected.
    
    Args:
        tokens (List[str]): List of Android registration tokens.
        camera_id (int): ID of the camera that detected fire.
    """
    if not tokens:
        return

    app = init_firebase()
    if not app:
        return

    success_count = 0
    failure_count = 0

    # Get camera label from database
    camera_label = await get_camera_label(camera_id)

    # Send notification to each token individually
    for token in tokens:
        # Build notification
        body_text = f"Fire detected at {camera_label}!"
        
        notification_obj = messaging.Notification(
            title="⚠️ FIRE ALERT!",
            body=body_text,
        )
        
        message = messaging.Message(
            notification=notification_obj,
            data={
                "camera_id": str(camera_id),
                "event_type": "fire_alert",
            },
            token=token,
        )
        
        try:
            # Send individual message
            await asyncio.to_thread(messaging.send, message)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to send to token {token[:20]}...: {e}")
            failure_count += 1
    
    print(f"✅ Successfully sent {success_count} notifications. ❌ {failure_count} failed.")

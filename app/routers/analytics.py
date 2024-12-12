from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.connection import get_db
from app.models import Analytics, User
from app.dependencies import get_current_user
import json

router = APIRouter()

@router.get("/analytic")
async def get_analytics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analytics = db.query(Analytics).filter(Analytics.user_id == current_user.user_id).first()
    if not analytics:
        return Response(
            content=json.dumps({"detail": "Analytics not found"}),
            status_code=404,
            media_type="application/json"
        )
    return Response(content=json.dumps({
        "i_scanned": {
            "total": analytics.total_i_scanned or 0,
            "successful": analytics.successful_i_scanned or 0,
            "failed": analytics.failed_i_scanned or 0,
            "last_time": analytics.last_time_i_scanned.isoformat() or 0
        },
        "whos_scanned_me": {
            "total": analytics.total_whos_scanned_me or 0,
            "successful": analytics.successful_whos_scanned_me or 0,
            "failed": analytics.failed_whos_scanned_me or 0,
            "last_time": analytics.last_time_whos_scanned_me.isoformat() or 0
        }
    }), status_code=200, media_type="application/json")
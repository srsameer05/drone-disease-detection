from fastapi import APIRouter
from dashboard.services.telemetry import telemetry_service

router = APIRouter()

@router.get("/api/telemetry")
async def get_telemetry():
    return telemetry_service.get_snapshot()

import threading
import time
from pymavlink import mavutil

class TelemetryService:
    def __init__(self, connection_string="udp:127.0.0.1:14551"):
        self.connection_string = connection_string
        self.active = False
        self._thread = None
        
        self.connected = False
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.battery = 0.0
        self.heading = 0.0
        self.speed = 0.0

    def start(self):
        if self.active:
            return
        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False
        self.connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self):
        while self.active:
            try:
                # Disable MAVLink stream auto-request and wait for heartbeat
                master = mavutil.mavlink_connection(self.connection_string)
                print(f"[TelemetryService] Waiting for heartbeat on {self.connection_string}...")
                self.connected = False
                master.wait_heartbeat(timeout=5)
                print(f"[TelemetryService] Heartbeat received!")
                self.connected = True
                
                # Request data stream
                master.mav.request_data_stream_send(
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    4, # rate in Hz
                    1 # start
                )
                
                while self.active:
                    msg = master.recv_match(
                        type=['GLOBAL_POSITION_INT', 'BATTERY_STATUS', 'VFR_HUD'], 
                        blocking=True, 
                        timeout=1.0
                    )
                    if not msg:
                        continue
                        
                    msg_type = msg.get_type()
                    if msg_type == 'GLOBAL_POSITION_INT':
                        self.lat = msg.lat / 1e7
                        self.lon = msg.lon / 1e7
                        self.alt = msg.relative_alt / 1000.0
                        print(f"[TelemetryService] GPS: {self.lat}, {self.lon}")
                    elif msg_type == 'BATTERY_STATUS':
                        if msg.battery_remaining != -1:
                            self.battery = float(msg.battery_remaining)
                    elif msg_type == 'VFR_HUD':
                        self.heading = float(msg.heading)
                        self.speed = float(msg.groundspeed)

            except Exception as e:
                print(f"[TelemetryService] Connection error: {e}")
                self.connected = False
                time.sleep(2)

    def get_snapshot(self):
        return {
            "connected": self.connected,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "battery": self.battery,
            "heading": self.heading,
            "speed": self.speed
        }

telemetry_service = TelemetryService()

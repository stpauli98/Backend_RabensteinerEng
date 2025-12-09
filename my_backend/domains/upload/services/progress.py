"""
Progress Tracker for Upload Domain
Real-time progress tracking with per-step ETA calculation
"""
import time
from typing import Dict, Optional, Any

from domains.upload.config import PHASE_TIME_PER_MB, EMIT_INTERVAL


class ProgressTracker:
    """
    Progress tracking sa ETA za SVE faze procesiranja.

    ETA se računa na osnovu:
    - Veličine fajla (bytes) za procjenu ukupnog vremena
    - Broja redova kada postane poznat
    - Real-time mjerenja za streaming fazu

    PAYLOAD koji se šalje na frontend:
    {
        'uploadId': str,           # ID uploada za WebSocket room
        'step': str,               # Naziv trenutne faze
        'progress': int,           # Progress percentage (0-100)
        'message': str,            # Poruka za prikaz korisniku
        'status': str,             # Status (processing, completed, error)
        'currentStep': int,        # Redni broj trenutnog koraka
        'totalSteps': int,         # Ukupan broj koraka
        'eta': int,                # ETA u sekundama
        'etaFormatted': str        # ETA formatiran (npr. "2m 30s")
    }
    """

    def __init__(self, upload_id: str, socketio=None, file_size_bytes: int = 0):
        self.upload_id = upload_id
        self.socketio = socketio
        self.start_time = time.time()
        self.phase_start_times: Dict[str, float] = {}
        self.phase_durations: Dict[str, float] = {}

        # File size za procjenu ETA
        self.file_size_bytes = file_size_bytes
        self.file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 1

        # Procijenjeno ukupno vrijeme na osnovu veličine fajla
        self.estimated_total_time = self._estimate_total_time()

        self.last_emit_time = 0
        self.emit_interval = EMIT_INTERVAL

        # Step tracking za frontend
        self.current_step = 0
        self.total_steps = 0

        # Broj redova (postavlja se nakon parsiranja)
        self.total_rows = 0

    def _estimate_total_time(self) -> float:
        """Procijeni ukupno vrijeme na osnovu veličine fajla"""
        total = 0
        for phase, time_per_mb in PHASE_TIME_PER_MB.items():
            total += time_per_mb * self.file_size_mb
        return max(total, 2.0)  # Minimum 2 sekunde

    def set_total_rows(self, rows: int) -> None:
        """Postavi broj redova i ažuriraj procjenu vremena"""
        self.total_rows = rows
        streaming_time = rows * 0.00005
        self.estimated_total_time = (
            PHASE_TIME_PER_MB['validation'] * self.file_size_mb +
            PHASE_TIME_PER_MB['parsing'] * self.file_size_mb +
            PHASE_TIME_PER_MB['datetime'] * self.file_size_mb +
            PHASE_TIME_PER_MB['utc'] * self.file_size_mb +
            PHASE_TIME_PER_MB['build'] * self.file_size_mb +
            streaming_time
        )

    def start_phase(self, phase_name: str) -> None:
        """Označi početak nove faze procesiranja"""
        self.phase_start_times[phase_name] = time.time()

    def end_phase(self, phase_name: str) -> None:
        """Završi fazu i snimi stvarno vrijeme trajanja"""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations[phase_name] = duration

    def calculate_eta_for_progress(self, current_progress: float) -> int:
        """
        Izračunaj ETA na osnovu trenutnog progresa i procijenjenog ukupnog vremena.
        """
        if current_progress <= 0:
            return int(self.estimated_total_time)

        if current_progress >= 100:
            return 0

        elapsed = time.time() - self.start_time
        remaining_progress = 100 - current_progress

        # Ako je proteklo više od 1 sekunde, koristi linearnu ekstrapolaciju
        if elapsed > 1.0:
            time_per_percent = elapsed / current_progress
            eta_seconds = int(remaining_progress * time_per_percent)
        else:
            # Za brze faze koristi procijenjeno vrijeme bazirano na file size
            eta_seconds = int(self.estimated_total_time * remaining_progress / 100)

        # Ograniči na razumne vrijednosti
        return max(0, min(eta_seconds, 3600))  # Max 1 sat

    def emit(
        self,
        step: str,
        progress: float,
        message_key: str,
        eta_seconds: Optional[int] = None,
        force: bool = False,
        processed_rows: Optional[int] = None,
        message_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Šalje progress update preko WebSocket-a sa ključem za prevod.
        """
        current_time = time.time()

        # Rate limiting
        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        payload = {
            'uploadId': self.upload_id,
            'step': step,
            'progress': int(progress),
            'messageKey': message_key,
            'status': 'processing' if step not in ['complete', 'error'] else ('completed' if step == 'complete' else 'error')
        }

        # Dodaj parametre za poruku ako postoje
        if message_params:
            payload['messageParams'] = message_params

        # Dodaj currentStep i totalSteps ako su postavljeni
        if self.total_steps > 0:
            payload['currentStep'] = self.current_step
            payload['totalSteps'] = self.total_steps

        # Dodaj totalRows i processedRows ako su dostupni
        if self.total_rows > 0:
            payload['totalRows'] = self.total_rows
        if processed_rows is not None:
            payload['processedRows'] = processed_rows

        # Dodaj ETA
        if eta_seconds is not None:
            payload['eta'] = eta_seconds
            payload['etaFormatted'] = self.format_time(eta_seconds)
        else:
            progress_eta = self.calculate_eta_for_progress(progress)
            payload['eta'] = progress_eta
            payload['etaFormatted'] = self.format_time(progress_eta)

        try:
            if self.socketio:
                self.socketio.emit('upload_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time
        except Exception:
            pass

    @staticmethod
    def format_time(seconds: Optional[int]) -> str:
        """Formatira sekunde u čitljiv format"""
        if seconds is None:
            return "Procjenjujem..."
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

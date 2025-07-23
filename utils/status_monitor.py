#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo do Monitor de Status - Sistema de Monitoramento de Buffet
"""

import os
import threading
import time
from datetime import datetime

class StatusMonitor:
    """
    Gerencia e exibe o status das câmeras em uma tabela no console.
    A exibição é atualizada a cada 5 segundos em uma thread separada.
    """
    
    # Cores ANSI
    COLORS = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }
    
    STATUS_MAP = {
        "Online": {"text": "Online", "color": "green"},
        "Offline": {"text": "Offline", "color": "red"},
    }
    
    def __init__(self, camera_ids):
        """
        Inicializa o monitor de status.
        
        Args:
            camera_ids: Uma lista dos IDs de todas as câmeras a serem monitoradas.
        """
        self.camera_statuses = {
            cam_id: {"status": "Offline", "connecting": True} 
            for cam_id in camera_ids
        }
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self._periodic_refresh, daemon=True)

    def start(self):
        """Inicia a thread de atualização periódica."""
        self.running = True
        self.thread.start()

    def stop(self):
        """Para a thread de atualização periódica."""
        self.running = False
        # Opcional: esperar a thread terminar
        if self.thread.is_alive():
            self.thread.join()

    def update_status(self, camera_id, status=None, connecting=None):
        """
        Atualiza o status de uma câmera.
        
        Args:
            camera_id: O ID da câmera a ser atualizada.
            status: O novo status ('Online', 'Offline').
            connecting: O novo estado de conexão (True/False).
        """
        with self.lock:
            if camera_id in self.camera_statuses:
                if status is not None:
                    self.camera_statuses[camera_id]["status"] = status
                if connecting is not None:
                    self.camera_statuses[camera_id]["connecting"] = connecting

    def _periodic_refresh(self):
        """Loop principal da thread que reimprime a tabela a cada 5 segundos."""
        while self.running:
            with self.lock:
                self._print_status_table()
            time.sleep(5)

    def _print_status_table(self):
        """
        Limpa o console e imprime a tabela de status atualizada.
        O método é privado, pois só deve ser chamado internamente.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        timestamp = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        header = f"--- Status das Câmeras [{timestamp}] ---"
        print(f"RÁSCAL 30 anos - Foodvision v1")
        print(f"Monitor de buffet")
        print(self.COLORS["bold"] + header + self.COLORS["end"])
        
        # Tabela
        print(f"| {'Câmera':<10} | {'Status':<10} | {'Connecting':<12} |")
        print(f"|{'-'*12}|{'-'*12}|{'-'*14}|")
        
        # Ordena por nome da câmera para uma exibição consistente
        sorted_cameras = sorted(self.camera_statuses.items())

        for cam_id, cam_data in sorted_cameras:
            # Status (Online/Offline)
            status_key = cam_data["status"]
            status_info = self.STATUS_MAP.get(status_key, {"text": "Unknown", "color": "red"})
            status_color = self.COLORS.get(status_info["color"], "")
            status_text = status_info["text"]
            
            # Connecting (True/False)
            connecting_val = cam_data["connecting"]
            connecting_text = str(connecting_val)
            connecting_color = self.COLORS['yellow'] if connecting_val else ""
            
            print(f"| {cam_id:<10} | {status_color}{status_text:<10}{self.COLORS['end']} | {connecting_color}{connecting_text:<12}{self.COLORS['end']} |")
        
        print("-" * (len(header) + 10)) 
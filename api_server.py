#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de API Server - Sistema de Monitoramento de Buffet
Fornece uma API para encerrar o sistema e enviar o arquivo buffet_data.json.
"""

import json
import logging
import os
import threading
import time
from flask import Flask, jsonify, send_file, Response
import cv2
from pathlib import Path

class APIServer:
    """
    Classe responsável por fornecer uma API HTTP para o sistema de monitoramento.
    """
    
    def __init__(self, camera_threads, camera_info, host='0.0.0.0', port=3320):
        """
        Inicializa o servidor de API.
        
        Args:
            camera_threads: Dicionário com as instâncias das threads de câmera.
            camera_info: Dicionário com informações de configuração das câmeras.
            host: Host para o servidor Flask.
            port: Porta para o servidor Flask.
        """
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.camera_threads = camera_threads
        self.camera_info = camera_info
        self.app = Flask(__name__)
        self.running = False
        self.server_thread = None
        
        # Configurar rotas
        self.setup_routes()
        
        self.logger.info(f"Servidor de API inicializado em {host}:{port}")
    
    def setup_routes(self):
        """
        Configura as rotas da API.
        """
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Retorna o status geral do sistema e das câmeras."""
            try:
                camera_statuses = {}
                for cam_id, cam_thread in self.camera_threads.items():
                    camera_statuses[cam_id] = {
                        "name": self.camera_info.get(cam_id, {}).get("name", cam_id),
                        "is_alive": cam_thread.is_alive(),
                        "running": cam_thread.running,
                        "frame_count": cam_thread.frame_count
                    }
                
                return jsonify({
                    "system_status": "running",
                    "camera_threads": len(self.camera_threads),
                    "cameras": camera_statuses
                })
            except Exception as e:
                self.logger.error(f"Erro ao obter status: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/cameras/<camera_id>/frame.jpg', methods=['GET'])
        def get_camera_frame(camera_id):
            """Retorna o último frame anotado de uma câmera como uma imagem JPEG."""
            if camera_id not in self.camera_threads:
                return jsonify({"error": "Camera ID not found"}), 404

            cam_thread = self.camera_threads[camera_id]
            frame = cam_thread.get_annotated_frame()

            if frame is None:
                return jsonify({"error": "Frame not available"}), 404

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    return jsonify({"error": "Failed to encode frame"}), 500
                
                return Response(buffer.tobytes(), mimetype='image/jpeg')
            except Exception as e:
                self.logger.error(f"Erro ao codificar frame para {camera_id}: {e}", exc_info=True)
                return jsonify({"error": "Frame encoding error"}), 500

    def run(self):
        """
        Inicia o servidor Flask em uma thread separada.
        """
        if self.running:
            self.logger.warning("Servidor já está em execução")
            return
        
        self.server_thread = threading.Thread(target=self.app.run, kwargs={'host': self.host, 'port': self.port})
        self.server_thread.daemon = True
        self.server_thread.start()
        self.running = True
        
        self.logger.info(f"Servidor API iniciado em http://{self.host}:{self.port}")
    
    def stop(self):
        """
        Para o servidor Flask. A thread daemon será encerrada com a aplicação.
        """
        if not self.running:
            return
            
        self.logger.info("Parando servidor API...")
        # A thread do servidor Flask é daemon, então ela terminará quando o app principal sair.
        # Uma parada mais graciosa poderia ser implementada aqui se necessário.
        self.running = False
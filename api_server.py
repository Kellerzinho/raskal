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
from flask import Flask, jsonify, send_file
from pathlib import Path

class APIServer:
    """
    Classe responsável por fornecer uma API HTTP para o sistema de monitoramento.
    """
    
    def __init__(self, host='0.0.0.0', port=3320, data_file="buffet_data.json", system_instance=None):
        """
        Inicializa o servidor de API.
        
        Args:
            host: Host para o servidor Flask
            port: Porta para o servidor Flask
            data_file: Caminho para o arquivo JSON com os dados
            system_instance: Instância do sistema para encerramento
        """
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.data_file = Path(data_file)
        self.system_instance = system_instance
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
        @self.app.route('/api/data', methods=['GET'])
        def get_data():
            try:
                # Verificar se o arquivo existe
                if not self.data_file.exists():
                    self.logger.warning(f"Arquivo {self.data_file} não encontrado")
                    return jsonify({"error": "Data file not found"}), 404
                
                # Enviar o arquivo
                self.logger.info(f"Enviando arquivo {self.data_file}")
                
                # Iniciar thread para encerrar o sistema após enviar o arquivo
                if self.system_instance is not None:
                    threading.Thread(target=self._shutdown_system, daemon=True).start()
                
                return send_file(self.data_file, mimetype='application/json', as_attachment=True)
                
            except Exception as e:
                self.logger.error(f"Erro ao processar requisição: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            try:
                # Verificar se o arquivo existe e tem dados
                if not self.data_file.exists():
                    return jsonify({
                        "status": "running",
                        "data_file": False,
                        "message": "Sistema em execução, arquivo de dados ainda não criado"
                    })
                
                # Verificar se o arquivo tem dados
                try:
                    with open(self.data_file, 'r') as f:
                        data = json.load(f)
                        data_count = sum(len(cameras) for cameras in data.values())
                except (json.JSONDecodeError, IOError):
                    data_count = 0
                
                # Retornar status
                return jsonify({
                    "status": "running",
                    "data_file": True,
                    "data_count": data_count,
                    "message": f"Sistema em execução, {data_count} itens monitorados"
                })
                
            except Exception as e:
                self.logger.error(f"Erro ao obter status: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _shutdown_system(self):
        """
        Encerra o sistema após um pequeno delay para garantir que a resposta HTTP seja enviada.
        """
        self.logger.info("Preparando para encerrar o sistema...")
        
        # Aguardar um momento para garantir que a resposta HTTP seja enviada
        time.sleep(2)
        
        # Encerrar o sistema
        if self.system_instance and hasattr(self.system_instance, 'stop'):
            self.logger.info("Encerrando o sistema de monitoramento...")
            self.system_instance.stop()
        else:
            self.logger.warning("Não foi possível encerrar o sistema")
    
    def start(self):
        """
        Inicia o servidor Flask em uma thread separada.
        """
        if self.running:
            self.logger.warning("Servidor já está em execução")
            return
        
        def run_flask():
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_flask, name="APIServerThread")
        self.server_thread.daemon = True
        self.server_thread.start()
        self.running = True
        
        self.logger.info(f"Servidor API iniciado em http://{self.host}:{self.port}")
    
    def stop(self):
        """
        Para o servidor Flask.
        """
        if not self.running:
            return
            
        self.logger.info("Parando servidor API...")
        self.running = False
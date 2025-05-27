#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Conexão com Câmeras - Sistema de Monitoramento de Buffet
Responsável por estabelecer e gerenciar conexões com câmeras ESP32Cam,
gerenciar streams HTTP e garantir a estabilidade do recebimento de imagens.
"""

import logging
import requests
import threading
import time
import cv2
import numpy as np
from urllib.parse import urlparse
from pathlib import Path
import json


class CameraConnection:
    """
    Classe para gerenciar a conexão com uma única câmera ESP32Cam.
    """
    
    def __init__(self, camera_config):
        """
        Inicializa a conexão com uma câmera ESP32Cam.
        
        Args:
            camera_config: Dicionário com a configuração da câmera
        """
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_config["id"]
        self.ip = camera_config["ip"]
        self.port = camera_config["port"]
        self.location = camera_config["location"]
        self.max_fps = camera_config["max_fps"]
        
        # URL para a stream HTTP
        #"http://admin:admin@127.0.0.1:8100/Mjpeg/9"
        self.stream_url = f"http://admin:admin@{self.ip}/Mjpeg/{self.port}"
        
        # Status da câmera
        self.connection_tested = False
        
        self.logger.debug(f"Câmera {self.camera_id} inicializada - IP: {self.ip}, Porta: {self.port}")
    
    
    def try_connect_to_stream(self, timeout=40):
        """
        Tenta conectar-se à stream HTTP da câmera.
        
        Args:
            timeout: Tempo limite para a tentativa de conexão (segundos)
            
        Returns:
            bool: True se a conexão for bem-sucedida, False caso contrário
        """
        try:
            # Tentar abrir a stream usando OpenCV
            cap = cv2.VideoCapture(self.stream_url)
            
            # Configurar timeout
            start_time = time.time()
            
            # Tentar ler frames até o timeout
            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                
                if ret:
                    # Se conseguir ler um frame, a conexão está funcionando
                    self.logger.info(f"Conexão com stream da câmera {self.camera_id} estabelecida com sucesso")
                    cap.release()  # Liberar recursos
                    return True
                
                time.sleep(0.1)
            
            # Se chegou aqui, o timeout foi atingido
            self.logger.warning(f"Timeout ao tentar conectar à stream da câmera {self.camera_id}")
            cap.release()
            return False
            
        except Exception as e:
            self.logger.exception(f"Erro ao tentar conectar à stream da câmera {self.camera_id}: {e}")
            return False
    
    def get_status(self):
        """
        Retorna o status atual da câmera.
        
        Returns:
            dict: Dicionário com informações de status da câmera
        """
        return {
            "id": self.camera_id,
            "ip": self.ip,
            "port": self.port,
            "location": self.location,
            "connection_tested": self.connection_tested,
            "is_connected": self.is_connected
        }


# Função para teste simples do módulo quando executado diretamente
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("=== Teste do Gerenciador de Câmeras ===")
    
    # Inicializar gerenciador de câmeras
    manager = CameraManager("cameras.json")
    
    # Testar conexão com todas as câmeras
    print("\nTestando conexão com todas as câmeras...")
    results = manager.test_all_connections()
    
    # Exibir resultados dos testes
    print("\nResultados dos testes de conexão:")
    for camera_id, success in results.items():
        status = "CONECTADA" if success else "FALHA"
        print(f"Câmera {camera_id}: {status}")
    
    # Testar streams para câmeras conectadas
    print("\nTestando streams de câmeras conectadas...")
    stream_results = manager.test_all_streams()
    
    # Exibir resultados dos testes de stream
    print("\nResultados dos testes de stream:")
    for camera_id, success in stream_results.items():
        if results.get(camera_id, False):  # Verificar se a câmera está conectada
            status = "DISPONÍVEL" if success else "INDISPONÍVEL"
            print(f"Stream da câmera {camera_id}: {status}")
    
    # Exibir status detalhado de todas as câmeras
    print("\nStatus detalhado das câmeras:")
    all_status = manager.get_all_status()
    for camera_id, status in all_status.items():
        print(f"Câmera {camera_id}:")
        for key, value in status.items():
            print(f"  - {key}: {value}")
    
    print("\n=== Teste finalizado ===")
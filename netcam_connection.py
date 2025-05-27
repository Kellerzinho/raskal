#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Conexão com NetCam Studio X - Sistema de Monitoramento de Buffet
Responsável por estabelecer conexões com streams do NetCam Studio X.
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


class NetCamStudioConnection:
    """
    Classe para gerenciar a conexão com NetCam Studio X.
    """
    
    def __init__(self, camera_config):
        """
        Inicializa a conexão com NetCam Studio X.
        
        Args:
            camera_config: Dicionário com a configuração da câmera
        """
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_config["id"]
        self.ip = camera_config["ip"]
        self.port = camera_config.get("port", 8100)  # Porta padrão do NetCam Studio X
        self.source_id = camera_config.get("source_id", 0)  # ID da source no NetCam Studio X
        self.username = camera_config.get("username", "admin")
        self.password = camera_config.get("password", "admin")
        self.location = camera_config.get("location", "")
        self.max_fps = camera_config.get("max_fps", 15)
        
        # URLs possíveis para NetCam Studio X
        self.stream_formats = {
            # Formato MJPEG - mais comum e estável
            "mjpeg": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/mjpeg/{self.source_id}",
            
            # Formato alternativo sem autenticação
            "mjpeg_no_auth": f"http://{self.ip}:{self.port}/mjpeg/{self.source_id}",
            
            # Formato com parâmetros adicionais
            "mjpeg_params": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/mjpeg/{self.source_id}?fps={self.max_fps}",
            
            # Formato de snapshot (para teste)
            "snapshot": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/snapshot/{self.source_id}",
            
            # Formato alternativo usado por algumas versões
            "cam": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/cam/{self.source_id}/mjpeg",
        }
        
        self.active_stream_url = None
        self.connection_tested = False
        self.is_connected = False
        
        self.logger.debug(f"NetCam Studio X {self.camera_id} inicializada - {self.ip}:{self.port}, Source: {self.source_id}")
    
    def test_connection(self, timeout=10):
        """
        Testa a conexão HTTP básica com o NetCam Studio X.
        
        Args:
            timeout: Tempo limite para a tentativa de conexão
            
        Returns:
            bool: True se a conexão for bem-sucedida
        """
        self.logger.info(f"Testando conexão com NetCam Studio X {self.camera_id}...")
        
        # Testar conexão básica primeiro
        test_url = f"http://{self.ip}:{self.port}"
        
        try:
            response = requests.get(test_url, timeout=timeout)
            if response.status_code == 200:
                self.logger.info(f"NetCam Studio X {self.camera_id} respondendo na porta {self.port}")
                self.connection_tested = True
                return True
            else:
                self.logger.warning(f"NetCam Studio X {self.camera_id} retornou status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout ao conectar com NetCam Studio X {self.camera_id}")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Erro de conexão com NetCam Studio X {self.camera_id}")
            return False
        except Exception as e:
            self.logger.error(f"Erro inesperado ao testar NetCam Studio X {self.camera_id}: {e}")
            return False
    
    def find_working_stream_url(self, timeout=30):
        """
        Tenta diferentes URLs de stream até encontrar uma que funcione.
        
        Args:
            timeout: Tempo limite para cada tentativa
            
        Returns:
            str: URL do stream que funciona, ou None se nenhuma funcionar
        """
        self.logger.info(f"Procurando URL de stream funcionando para {self.camera_id}...")
        
        for format_name, url in self.stream_formats.items():
            self.logger.debug(f"Testando formato {format_name}: {url}")
            
            try:
                # Testar com OpenCV
                cap = cv2.VideoCapture(url)
                
                if not cap.isOpened():
                    self.logger.debug(f"Não foi possível abrir {format_name}")
                    cap.release()
                    continue
                
                # Tentar ler alguns frames para confirmar que funciona
                success_count = 0
                for i in range(5):  # Testar 5 frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success_count += 1
                    time.sleep(0.1)
                
                cap.release()
                
                if success_count >= 2:  # Pelo menos 2 frames lidos com sucesso
                    self.logger.info(f"Stream funcionando com formato {format_name}")
                    self.active_stream_url = url
                    self.is_connected = True
                    return url
                else:
                    self.logger.debug(f"Formato {format_name} não retornou frames válidos")
                    
            except Exception as e:
                self.logger.debug(f"Erro ao testar formato {format_name}: {e}")
                continue
        
        self.logger.error(f"Nenhum formato de stream funcionou para {self.camera_id}")
        return None
    
    def try_connect_to_stream(self, timeout=40):
        """
        Tenta conectar-se à stream do NetCam Studio X.
        
        Args:
            timeout: Tempo limite para a tentativa de conexão
            
        Returns:
            bool: True se a conexão for bem-sucedida
        """
        # Primeiro testar a conexão básica
        if not self.test_connection():
            return False
        
        # Tentar encontrar uma URL de stream que funcione
        working_url = self.find_working_stream_url(timeout)
        
        if working_url:
            self.logger.info(f"Conexão com stream do NetCam Studio X {self.camera_id} estabelecida")
            return True
        else:
            self.logger.error(f"Falha ao estabelecer conexão com stream do NetCam Studio X {self.camera_id}")
            return False
    
    def get_stream_capture(self):
        """
        Retorna um objeto VideoCapture configurado para o stream.
        
        Returns:
            cv2.VideoCapture: Objeto configurado ou None se não houver stream ativo
        """
        if not self.active_stream_url:
            self.logger.error(f"Nenhum stream ativo para {self.camera_id}")
            return None
        
        cap = cv2.VideoCapture(self.active_stream_url)
        
        if cap.isOpened():
            # Configurar propriedades do capture se necessário
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer para diminuir latência
            return cap
        else:
            self.logger.error(f"Não foi possível abrir capture para {self.camera_id}")
            return None
    
    def test_snapshot(self):
        """
        Testa captura de snapshot para verificar se a câmera está funcionando.
        
        Returns:
            bool: True se conseguir capturar um snapshot
        """
        snapshot_url = self.stream_formats["snapshot"]
        
        try:
            response = requests.get(snapshot_url, timeout=10)
            if response.status_code == 200 and response.content:
                self.logger.info(f"Snapshot capturado com sucesso para {self.camera_id}")
                return True
            else:
                self.logger.warning(f"Falha ao capturar snapshot para {self.camera_id}: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao testar snapshot para {self.camera_id}: {e}")
            return False
    
    def get_status(self):
        """
        Retorna o status atual da conexão.
        
        Returns:
            dict: Dicionário com informações de status
        """
        return {
            "id": self.camera_id,
            "ip": self.ip,
            "port": self.port,
            "source_id": self.source_id,
            "location": self.location,
            "connection_tested": self.connection_tested,
            "is_connected": self.is_connected,
            "active_stream_url": self.active_stream_url,
            "username": self.username
        }


class NetCamStudioManager:
    """
    Gerenciador para múltiplas conexões NetCam Studio X.
    """
    
    def __init__(self, config_file="config/cameras.json"):
        """
        Inicializa o gerenciador.
        
        Args:
            config_file: Arquivo de configuração das câmeras
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.connections = {}
        self.load_camera_config()
    
    def load_camera_config(self):
        """
        Carrega a configuração das câmeras do arquivo JSON.
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            for camera_config in config.get("cameras", []):
                camera_id = camera_config["id"]
                connection = NetCamStudioConnection(camera_config)
                self.connections[camera_id] = connection
                
            self.logger.info(f"Carregadas {len(self.connections)} configurações de câmera")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração das câmeras: {e}")
    
    def test_all_connections(self):
        """
        Testa conexão com todas as câmeras.
        
        Returns:
            dict: Resultados dos testes por camera_id
        """
        results = {}
        
        for camera_id, connection in self.connections.items():
            results[camera_id] = connection.try_connect_to_stream()
            
        return results
    
    def get_connection(self, camera_id):
        """
        Obtém a conexão para uma câmera específica.
        
        Args:
            camera_id: ID da câmera
            
        Returns:
            NetCamStudioConnection: Conexão ou None se não encontrada
        """
        return self.connections.get(camera_id)
    
    def get_all_status(self):
        """
        Obtém o status de todas as conexões.
        
        Returns:
            dict: Status de todas as conexões
        """
        status = {}
        
        for camera_id, connection in self.connections.items():
            status[camera_id] = connection.get_status()
            
        return status


# Exemplo de configuração atualizada para cameras.json
EXAMPLE_CONFIG = {
    "cameras": [
        {
            "id": "cam0",
            "ip": "192.168.1.100",  # IP do computador rodando NetCam Studio X
            "port": 8100,           # Porta do web server do NetCam Studio X
            "source_id": 0,         # ID da source no NetCam Studio X (0, 1, 2, etc.)
            "username": "admin",    # Usuário configurado no NetCam Studio X
            "password": "admin",    # Senha configurada no NetCam Studio X
            "location": "Buffet Principal",
            "max_fps": 15
        },
        {
            "id": "cam1",
            "ip": "192.168.1.100",
            "port": 8100,
            "source_id": 1,
            "username": "admin",
            "password": "admin",
            "location": "Buffet Secundário",
            "max_fps": 15
        }
    ]
}


# Função de teste
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("=== Teste de Conexão NetCam Studio X ===")
    
    # Exemplo de uso direto
    test_config = {
        "id": "test_cam",
        "ip": "127.0.0.1",      # Substitua pelo IP do seu NetCam Studio X
        "port": 8100,
        "source_id": 0,
        "username": "admin",
        "password": "admin",
        "location": "Teste",
        "max_fps": 15
    }
    
    # Criar conexão
    connection = NetCamStudioConnection(test_config)
    
    # Testar conexão
    if connection.try_connect_to_stream():
        print("✓ Conexão estabelecida com sucesso!")
        print(f"URL ativa: {connection.active_stream_url}")
        
        # Testar captura de alguns frames
        cap = connection.get_stream_capture()
        if cap:
            print("Testando captura de frames...")
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"Frame {i+1}: {frame.shape}")
                else:
                    print(f"Falha ao capturar frame {i+1}")
                time.sleep(0.5)
            cap.release()
        
    else:
        print("✗ Falha na conexão")
        
    # Mostrar status
    status = connection.get_status()
    print(f"\nStatus da conexão:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n=== Teste finalizado ===")
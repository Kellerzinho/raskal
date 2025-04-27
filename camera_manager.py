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
        self.stream_url = f"http://{self.ip}:{self.port}/stream"
        
        # Status da câmera
        self.is_connected = False
        self.connection_tested = False
        
        self.logger.debug(f"Câmera {self.camera_id} inicializada - IP: {self.ip}, Porta: {self.port}")
    
    def test_connection(self, timeout=20):
        """
        Testa a conexão com a câmera usando uma requisição HTTP simples.
        
        Args:
            timeout: Tempo limite para a requisição (segundos)
            
        Returns:
            bool: True se a conexão for bem-sucedida, False caso contrário
        """
        self.logger.info(f"Testando conexão com a câmera {self.camera_id} ({self.ip}:{self.port})")
        
        try:
            # URL para acessar a página inicial da ESP32Cam (geralmente o status)
            status_url = f"http://{self.ip}:{self.port}/"
            
            # Realizar uma requisição GET com timeout
            response = requests.get(status_url, timeout=timeout)
            
            # Verificar se a requisição foi bem-sucedida (código 200)
            if response.status_code == 200:
                self.logger.info(f"Conexão com a câmera {self.camera_id} estabelecida com sucesso")
                self.connection_tested = True
                self.is_connected = True
                return True
            else:
                self.logger.warning(f"Conexão com a câmera {self.camera_id} falhou - Código: {response.status_code}")
                self.connection_tested = True
                self.is_connected = False
                return False
                
        except requests.exceptions.ConnectTimeout:
            self.logger.error(f"Timeout ao conectar com a câmera {self.camera_id}")
            self.connection_tested = True
            self.is_connected = False
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Erro de conexão com a câmera {self.camera_id} - Câmera não disponível")
            self.connection_tested = True
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.exception(f"Erro inesperado ao testar conexão com câmera {self.camera_id}: {e}")
            self.connection_tested = True
            self.is_connected = False
            return False
    
    def try_connect_to_stream(self, timeout=20):
        """
        Tenta conectar-se à stream HTTP da câmera.
        
        Args:
            timeout: Tempo limite para a tentativa de conexão (segundos)
            
        Returns:
            bool: True se a conexão for bem-sucedida, False caso contrário
        """
        if not self.is_connected:
            self.logger.warning(f"Não é possível conectar à stream da câmera {self.camera_id} - Câmera não conectada")
            return False
            
        self.logger.info(f"Tentando conectar à stream da câmera {self.camera_id}: {self.stream_url}")
        
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


class CameraManager:
    """
    Classe para gerenciar múltiplas câmeras ESP32Cam.
    """
    
    def __init__(self, config_path="config/cameras.json"):
        """
        Inicializa o gerenciador de câmeras.
        
        Args:
            config_path: Caminho para o arquivo de configuração das câmeras
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self.cameras = {}
        
        # Carregar configuração das câmeras
        self.load_config()
        
        self.logger.info(f"Gerenciador de câmeras inicializado com {len(self.cameras)} câmeras")
    
    def load_config(self):
        """
        Carrega a configuração das câmeras a partir do arquivo JSON.
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                
            for camera_config in config["cameras"]:
                camera_id = camera_config["id"]
                self.cameras[camera_id] = CameraConnection(camera_config)
                
            self.logger.info(f"Configuração de {len(self.cameras)} câmeras carregada com sucesso")
                
        except FileNotFoundError:
            self.logger.error(f"Arquivo de configuração não encontrado: {self.config_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Erro ao decodificar o arquivo JSON: {self.config_path}")
            raise
        except Exception as e:
            self.logger.exception(f"Erro inesperado ao carregar configuração das câmeras: {e}")
            raise
    
    def test_all_connections(self, timeout=20, update_monitor=None):
        """
        Testa a conexão com todas as câmeras configuradas.
        
        Args:
            timeout: Tempo limite para cada teste de conexão (segundos)
            update_monitor: Função de callback para atualizar o monitor de sistema (opcional)
            
        Returns:
            dict: Dicionário com os resultados dos testes para cada câmera
        """
        self.logger.info(f"Testando conexão com {len(self.cameras)} câmeras")
        results = {}
        
        for camera_id, camera in self.cameras.items():
            result = camera.test_connection(timeout=timeout)
            results[camera_id] = result
            
            # Atualizar monitor de sistema, se fornecido
            if update_monitor is not None:
                update_monitor(camera_id, "connection_tested", True)
                update_monitor(camera_id, "connection_established", result)
            
        # Contar câmeras conectadas com sucesso
        successful = sum(1 for result in results.values() if result)
        self.logger.info(f"Teste de conexão finalizado: {successful}/{len(results)} câmeras conectadas")
        
        return results
    
    def test_connection(self, camera_id, timeout=3, update_monitor=None):
        """
        Testa a conexão com uma câmera específica.
        
        Args:
            camera_id: ID da câmera a ser testada
            timeout: Tempo limite para o teste de conexão (segundos)
            update_monitor: Função de callback para atualizar o monitor de sistema (opcional)
            
        Returns:
            bool: True se a conexão for bem-sucedida, False caso contrário
        """
        if camera_id not in self.cameras:
            self.logger.warning(f"Câmera {camera_id} não encontrada")
            return False
            
        result = self.cameras[camera_id].test_connection(timeout=timeout)
        
        # Atualizar monitor de sistema, se fornecido
        if update_monitor is not None:
            update_monitor(camera_id, "connection_tested", True)
            update_monitor(camera_id, "connection_established", result)
            
        return result
    
    def test_all_streams(self, timeout=20, update_monitor=None):
        """
        Testa a conexão com as streams de todas as câmeras conectadas.
        
        Args:
            timeout: Tempo limite para cada teste de stream (segundos)
            update_monitor: Função de callback para atualizar o monitor de sistema (opcional)
            
        Returns:
            dict: Dicionário com os resultados dos testes para cada câmera
        """
        self.logger.info("Testando streams de todas as câmeras conectadas")
        results = {}
        
        for camera_id, camera in self.cameras.items():
            if camera.is_connected:
                result = camera.try_connect_to_stream(timeout=timeout)
                results[camera_id] = result
                
                # Atualizar monitor de sistema, se fornecido
                if update_monitor is not None:
                    update_monitor(camera_id, "model_processing", result)
            else:
                results[camera_id] = False
                self.logger.warning(f"Câmera {camera_id} não está conectada, stream não testado")
                
        # Contar streams conectados com sucesso
        successful = sum(1 for result in results.values() if result)
        self.logger.info(f"Teste de streams finalizado: {successful}/{len(results)} streams acessíveis")
        
        return results
    
    def test_stream(self, camera_id, timeout=5, update_monitor=None):
        """
        Testa a conexão com a stream de uma câmera específica.
        
        Args:
            camera_id: ID da câmera a ser testada
            timeout: Tempo limite para o teste de stream (segundos)
            update_monitor: Função de callback para atualizar o monitor de sistema (opcional)
            
        Returns:
            bool: True se a conexão for bem-sucedida, False caso contrário
        """
        if camera_id not in self.cameras:
            self.logger.warning(f"Câmera {camera_id} não encontrada")
            return False
            
        camera = self.cameras[camera_id]
        
        if not camera.is_connected:
            self.logger.warning(f"Câmera {camera_id} não está conectada, stream não testado")
            return False
            
        result = camera.try_connect_to_stream(timeout=timeout)
        
        # Atualizar monitor de sistema, se fornecido
        if update_monitor is not None:
            update_monitor(camera_id, "model_processing", result)
            
        return result
    
    def get_camera_status(self, camera_id):
        """
        Obtém o status de uma câmera específica.
        
        Args:
            camera_id: ID da câmera
            
        Returns:
            dict: Dicionário com o status da câmera ou None se a câmera não existir
        """
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_status()
        else:
            self.logger.warning(f"Câmera {camera_id} não encontrada")
            return None
    
    def get_all_status(self):
        """
        Obtém o status de todas as câmeras.
        
        Returns:
            dict: Dicionário com o status de todas as câmeras
        """
        return {camera_id: camera.get_status() for camera_id, camera in self.cameras.items()}
    
    def get_camera(self, camera_id):
        """
        Obtém uma instância de câmera específica.
        
        Args:
            camera_id: ID da câmera
            
        Returns:
            CameraConnection: Instância da câmera ou None se a câmera não existir
        """
        return self.cameras.get(camera_id)
    
    def update_system_monitor(self, system_monitor):
        """
        Atualiza as informações no monitor de sistema com o status atual das câmeras.
        
        Args:
            system_monitor: Instância de SystemMonitor para atualizar
        """
        if system_monitor is None:
            return
            
        for camera_id, camera in self.cameras.items():
            status = camera.get_status()
            system_monitor.camera_status[camera_id]["connection_tested"] = status["connection_tested"]
            system_monitor.camera_status[camera_id]["connection_established"] = status["is_connected"]
        
        self.logger.debug("Status do monitor de sistema atualizado com informações das câmeras")


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
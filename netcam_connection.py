#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Conexão com NetCam Studio X - Sistema de Monitoramento de Buffet
Versão corrigida baseada na documentação oficial do NetCam Studio X.
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
        
        # URLs corretas baseadas na documentação encontrada
        self.stream_formats = {
            # Formato MJPEG correto (com M maiúsculo)
            "mjpeg": f"http://{self.ip}:{self.port}/Mjpeg/{self.source_id}",
            
            # Formato com autenticação básica
            "mjpeg_auth": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/Mjpeg/{self.source_id}",
            
            # Formato Live (H.264 stream)
            "live": f"http://{self.ip}:{self.port}/Live/{self.source_id}",
            
            # Formato Live com autenticação
            "live_auth": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/Live/{self.source_id}",
            
            # Snapshot para teste
            "snapshot": f"http://{self.ip}:{self.port}/Snapshot/{self.source_id}",
            
            # Snapshot com autenticação
            "snapshot_auth": f"http://{self.username}:{self.password}@{self.ip}:{self.port}/Snapshot/{self.source_id}",
            
            # Formato com parâmetros (se necessário)
            "mjpeg_params": f"http://{self.ip}:{self.port}/Mjpeg/{self.source_id}?width=640&height=480",
        }
        
        self.active_stream_url = None
        self.connection_tested = False
        self.is_connected = False
        self.web_server_enabled = False
        
        self.logger.debug(f"NetCam Studio X {self.camera_id} inicializada - {self.ip}:{self.port}, Source: {self.source_id}")
    
    def test_web_server(self, timeout=10):
        """
        Testa se o web server do NetCam Studio X está habilitado e funcionando.
        
        Args:
            timeout: Tempo limite para a tentativa de conexão
            
        Returns:
            bool: True se o web server estiver funcionando
        """
        self.logger.info(f"Testando web server do NetCam Studio X {self.camera_id}...")
        
        # URLs de teste para verificar se o web server está ativo
        test_urls = [
            f"http://{self.ip}:{self.port}",
            f"http://{self.ip}:{self.port}/",
            f"http://{self.ip}:{self.port}/api",
        ]
        
        for test_url in test_urls:
            try:
                self.logger.debug(f"Testando URL: {test_url}")
                response = requests.get(test_url, timeout=timeout)
                
                # NetCam Studio X pode retornar diferentes códigos de status
                if response.status_code in [200, 401, 403]:  # 401/403 indica que o servidor está ativo mas precisa auth
                    self.logger.info(f"Web server NetCam Studio X {self.camera_id} está ativo")
                    self.connection_tested = True
                    self.web_server_enabled = True
                    
                    if response.status_code == 401:
                        self.logger.info(f"Servidor requer autenticação (HTTP 401)")
                    elif response.status_code == 403:
                        self.logger.info(f"Acesso negado - verifique permissões (HTTP 403)")
                    
                    return True
                else:
                    self.logger.debug(f"Servidor retornou status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                self.logger.debug(f"Timeout ao testar {test_url}")
                continue
            except requests.exceptions.ConnectionError:
                self.logger.debug(f"Erro de conexão ao testar {test_url}")
                continue
            except Exception as e:
                self.logger.debug(f"Erro inesperado ao testar {test_url}: {e}")
                continue
        
        self.logger.error(f"Web server do NetCam Studio X {self.camera_id} não está respondendo ou não está habilitado")
        self.logger.error(f"Verifique se:")
        self.logger.error(f"1. NetCam Studio X está executando")
        self.logger.error(f"2. Web Server está habilitado nas configurações")
        self.logger.error(f"3. A porta {self.port} está correta")
        self.logger.error(f"4. Não há firewall bloqueando a conexão")
        
        return False
    
    def test_snapshot(self):
        """
        Testa captura de snapshot antes de tentar o stream.
        
        Returns:
            bool: True se conseguir capturar um snapshot
        """
        self.logger.info(f"Testando snapshot para câmera {self.camera_id}...")
        
        # Testar diferentes formatos de snapshot
        snapshot_urls = [
            self.stream_formats["snapshot"],
            self.stream_formats["snapshot_auth"]
        ]
        
        for url in snapshot_urls:
            try:
                self.logger.debug(f"Testando snapshot URL: {url}")
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200 and len(response.content) > 1000:  # Assumir que uma imagem tem pelo menos 1KB
                    self.logger.info(f"Snapshot capturado com sucesso para {self.camera_id}")
                    return True
                elif response.status_code == 401:
                    self.logger.warning(f"Snapshot requer autenticação para {self.camera_id}")
                    continue
                else:
                    self.logger.debug(f"Snapshot falhou: {response.status_code}")
                    
            except Exception as e:
                self.logger.debug(f"Erro ao testar snapshot: {e}")
                continue
        
        self.logger.warning(f"Não foi possível capturar snapshot para {self.camera_id}")
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
        
        # Ordem de prioridade baseada na documentação encontrada
        priority_order = [
            "mjpeg",        # MJPEG sem auth (mais comum)
            "mjpeg_auth",   # MJPEG com auth
            "live",         # H.264 sem auth
            "live_auth",    # H.264 com auth
            "mjpeg_params", # MJPEG com parâmetros
            "snapshot",     # Snapshot sem auth
            "snapshot_auth" # Snapshot com auth
        ]
        
        for format_name in priority_order:
            url = self.stream_formats[format_name]
            self.logger.debug(f"Testando formato {format_name}: {url}")
            
            try:
                # Testar com OpenCV
                cap = cv2.VideoCapture(url)
                
                if not cap.isOpened():
                    self.logger.debug(f"OpenCV não conseguiu abrir {format_name}")
                    cap.release()
                    continue
                
                # Tentar ler alguns frames para confirmar que funciona
                success_count = 0
                test_frames = 3  # Reduzido para 3 frames
                
                for i in range(test_frames):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success_count += 1
                        self.logger.debug(f"Frame {i+1} lido com sucesso: {frame.shape}")
                    else:
                        self.logger.debug(f"Falha ao ler frame {i+1}")
                    time.sleep(0.2)  # Aguardar um pouco entre frames
                
                cap.release()
                
                if success_count >= 1:  # Pelo menos 1 frame válido
                    self.logger.info(f"Stream funcionando com formato {format_name}")
                    self.logger.info(f"URL ativa: {url}")
                    self.active_stream_url = url
                    self.is_connected = True
                    return url
                else:
                    self.logger.debug(f"Formato {format_name} não retornou frames válidos ({success_count}/{test_frames})")
                    
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
        self.logger.info(f"Iniciando conexão com NetCam Studio X {self.camera_id}")
        
        # Passo 1: Testar se o web server está funcionando
        if not self.test_web_server():
            return False
        
        # Passo 2: Testar snapshot (opcional, mas ajuda a diagnosticar)
        self.test_snapshot()
        
        # Passo 3: Tentar encontrar uma URL de stream que funcione
        working_url = self.find_working_stream_url(timeout)
        
        if working_url:
            self.logger.info(f"Conexão com stream do NetCam Studio X {self.camera_id} estabelecida com sucesso")
            return True
        else:
            self.logger.error(f"Falha ao estabelecer conexão com stream do NetCam Studio X {self.camera_id}")
            self.print_troubleshooting_tips()
            return False
    
    def print_troubleshooting_tips(self):
        """
        Imprime dicas de solução de problemas.
        """
        self.logger.error("=== DICAS DE SOLUÇÃO DE PROBLEMAS ===")
        self.logger.error("1. Verifique se o NetCam Studio X está executando")
        self.logger.error("2. Habilite o Web Server nas configurações:")
        self.logger.error("   - Abra NetCam Studio X")
        self.logger.error("   - Vá em Settings/Options")
        self.logger.error("   - Procure por 'Web Server' ou 'HTTP Server'")
        self.logger.error("   - Certifique-se de que está habilitado")
        self.logger.error(f"3. Verifique se a porta {self.port} está correta")
        self.logger.error(f"4. Verifique se o source_id {self.source_id} existe")
        self.logger.error("5. Teste manualmente no navegador:")
        self.logger.error(f"   - http://{self.ip}:{self.port}")
        self.logger.error(f"   - http://{self.ip}:{self.port}/Mjpeg/{self.source_id}")
        self.logger.error("6. Verifique firewall e antivírus")
        self.logger.error("7. Para versão gratuita: apenas 2 câmeras são suportadas para stream")
        self.logger.error("========================================")
    
    @property
    def stream_url(self):
        """
        Propriedade para compatibilidade com o código existente.
        
        Returns:
            str: URL ativa do stream ou None se não houver
        """
        return self.active_stream_url

    # CORREÇÃO: A definição do método abaixo estava incorretamente indentada
    # dentro de uma propriedade duplicada. Agora está no nível correto da classe.
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
            # Configurar propriedades do capture
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer para diminuir latência
            
            # Tentar configurar FPS se suportado
            try:
                cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            except:
                pass  # Nem todos os streams suportam configuração de FPS
                
            return cap
        else:
            self.logger.error(f"Não foi possível abrir capture para {self.camera_id}")
            return None
    
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
            "web_server_enabled": self.web_server_enabled,
            "is_connected": self.is_connected,
            "active_stream_url": self.active_stream_url,
            "username": self.username
        }


# Função de teste melhorada
def test_netcam_connection(ip="127.0.0.1", port=8100, source_id=0):
    """
    Função de teste independente para NetCam Studio X.
    
    Args:
        ip: IP do servidor NetCam Studio X
        port: Porta do web server
        source_id: ID da source/câmera
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("=== Teste de Conexão NetCam Studio X ===")
    print(f"Testando: {ip}:{port}, Source ID: {source_id}")
    
    # Configuração de teste
    test_config = {
        "id": "test_cam",
        "ip": ip,
        "port": port,
        "source_id": source_id,
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
            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Frame {i+1}: {frame.shape}")
                else:
                    print(f"✗ Falha ao capturar frame {i+1}")
                time.sleep(0.5)
            cap.release()
        
    else:
        print("✗ Falha na conexão")
        
    # Mostrar status
    status = connection.get_status()
    print(f"\n=== Status da Conexão ===")
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n=== Teste finalizado ===")


if __name__ == "__main__":
    # Teste com configurações padrão
    test_netcam_connection()
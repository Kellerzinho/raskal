#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Sincronização com API - Sistema de Monitoramento de Buffet
Responsável por enviar os dados processados para um dashboard externo.
"""

import os
import json
import time
import logging
import threading


class APIThread(threading.Thread):
    """
    Thread responsável por sincronizar os dados do arquivo buffet_data.json
    com o dashboard externo usando a API.
    """
    
    def __init__(self, api_url, sync_interval=5, token=None):
        """
        Inicializa a thread de sincronização com a API.
        
        Args:
            api_url: URL da API externa
            sync_interval: Intervalo entre sincronizações (segundos)
            token: Token de autenticação para a API (opcional)
        """
        super().__init__(name="APIThread")
        self.logger = logging.getLogger(__name__)
        self.api_url = api_url
        self.sync_interval = sync_interval
        self.token = token
        self.running = False
        self.data_file = "buffet_data.json"
        self.data_file_lock = threading.Lock()
        self.last_sync_time = 0
        
        # Importar o cliente de API externa
        from externalAPI import ExternalAPIClient
        self.api_client = ExternalAPIClient(
            api_url=self.api_url,
            auth_token=self.token
        )
        
        self.logger.debug(f"Thread de API inicializada com URL: {api_url}, Intervalo: {sync_interval}s")
    
    def run(self):
        """
        Função principal da thread.
        Sincroniza os dados com o dashboard em intervalos regulares.
        """
        self.logger.info("Thread de sincronização com API iniciada")
        self.running = True
        
        while self.running:
            try:
                # Verificar se é hora de sincronizar
                current_time = time.time()
                if current_time - self.last_sync_time >= self.sync_interval:
                    self.sync_data()
                    self.last_sync_time = current_time
                
                # Dormir por um curto período para não consumir CPU
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Erro durante sincronização com API: {e}")
                time.sleep(self.sync_interval)  # Esperar antes de tentar novamente
    
    def sync_data(self):
        """
        Sincroniza os dados do arquivo JSON com o dashboard.
        """
        try:
            # Verificar se o arquivo existe
            if not os.path.exists(self.data_file):
                self.logger.debug("Arquivo de dados não encontrado, pulando sincronização")
                return
            
            # Verificar se o arquivo está vazio
            if os.path.getsize(self.data_file) == 0:
                self.logger.debug("Arquivo de dados vazio, pulando sincronização")
                return
            
            # Carregar dados do arquivo JSON
            data = None
            with self.data_file_lock:
                try:
                    with open(self.data_file, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    self.logger.error(f"Erro ao decodificar {self.data_file}, pulando sincronização")
                    return
            
            # Verificar se há dados para enviar
            if not data:
                self.logger.debug("Sem dados para sincronizar")
                return
            
            # Enviar dados para a API externa
            success = self.api_client.send_data(data)
            
            if success:
                self.logger.info("Dados sincronizados com sucesso")
            else:
                self.logger.warning("Falha ao sincronizar dados")
                
        except Exception as e:
            self.logger.error(f"Erro durante sincronização dos dados: {e}")
    
    def stop(self):
        """
        Para a execução da thread.
        """
        self.logger.info("Parando thread de sincronização com API")
        self.running = False


# Função para teste simples do módulo quando executado diretamente
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("=== Teste do Módulo de Sincronização com API ===")
    
    # Criar uma instância da thread de API
    api_thread = APIThread(
        api_url="http://localhost:3300/api/replacements/updateReplacements",
        sync_interval=5
    )
    
    try:
        # Iniciar a thread
        api_thread.start()
        
        # Manter o programa rodando por 60 segundos
        print("Thread iniciada. Pressione Ctrl+C para encerrar...")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\nInterrupção de teclado detectada")
        
    finally:
        # Parar a thread
        api_thread.stop()
        api_thread.join(timeout=2.0)
        
        print("\n=== Teste finalizado ===")
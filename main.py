#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo Principal - Sistema de Monitoramento de Buffet
Responsável por coordenar o fluxo de processamento do sistema,
gerenciar threads/processos e controlar o ciclo de vida da aplicação.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


class LoggerManager:
    """
    Classe responsável por gerenciar a configuração do sistema de logging.
    Configura apenas o root logger com filter, formatter e handlers.
    """
    
    def __init__(self, log_dir="logs"):
        """
        Inicializa o gerenciador de logging.
        
        Args:
            log_dir: Diretório onde os logs serão armazenados
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar o root logger
        self.setup_root_logger()
        
        # Obter um logger para esta classe
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Sistema de logging inicializado")
    
    def setup_root_logger(self):
        """
        Configura o root logger com handlers, formatter e filter.
        """
        # Definir nome do arquivo de log com timestamp
        log_filename = f"buffet_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = self.log_dir / log_filename
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Limpar handlers existentes (caso já existam)
        if root_logger.handlers:
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
        
        # Criar console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Criar file handler para logs completos
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Definir formatter comum para todos os handlers
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Criar filter
        class PackageFilter(logging.Filter):
            def filter(self, record):
                # Exemplo: filtrar mensagens de dependências externas muito verbosas
                return not record.name.startswith("urllib3")
        
        # Adicionar filter apenas ao root logger
        root_logger.addFilter(PackageFilter())
        
        # Adicionar handlers ao root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)


class BuffetMonitoringSystem:
    """
    Classe principal do sistema de monitoramento de buffet.
    Coordena todos os módulos e gerencia o ciclo de vida da aplicação.
    """
    
    def __init__(self, config_path="config"):
        """
        Inicializa o sistema de monitoramento.
        
        Args:
            config_path: Caminho para o diretório de configurações
        """
        # Inicializar o gerenciador de logs primeiro
        self.logger_manager = LoggerManager()
        self.logger = logging.getLogger(__name__)
        
        self.config_path = Path(config_path)
        self.running = False
        self.cuda_available = False
        
        self.logger.info("Inicializando Sistema de Monitoramento de Buffet")
        
        # Carregar configurações
        self.load_config()
        
        # Verificar disponibilidade de CUDA
        self.check_cuda()
        
        # A importação e inicialização dos demais módulos será feita posteriormente
        self.logger.debug("Sistema inicializado com sucesso")
    
    def load_config(self):
        """
        Carrega arquivos de configuração do sistema.
        """
        try:
            # Carregar configuração das câmeras
            with open(self.config_path / "cameras.json", "r") as f:
                self.cameras_config = json.load(f)
                self.logger.debug(f"Configuração de câmeras carregada: {len(self.cameras_config['cameras'])} câmeras encontradas")
            
            # Aqui seriam carregadas outras configurações necessárias
            
        except FileNotFoundError as e:
            self.logger.error(f"Arquivo de configuração não encontrado: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar arquivo JSON: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.exception(f"Erro inesperado ao carregar configurações: {e}")
            sys.exit(1)
    
    def check_cuda(self):
        """
        Verifica a disponibilidade de aceleração CUDA e 
        permite escolher continuar ou não caso não esteja disponível.
        """
        self.logger.info("Verificando disponibilidade de aceleração CUDA...")
        
        try:
            # Tentar importar torch para verificar CUDA
            import torch
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                cuda_devices = torch.cuda.device_count()
                device_info = torch.cuda.get_device_name(0) if cuda_devices > 0 else "Unknown"
                self.logger.info(f"CUDA disponível! Dispositivos: {cuda_devices}, GPU: {device_info}")
            else:
                self.logger.warning("Aceleração CUDA não disponível!")
                
                # Perguntar ao usuário se deseja continuar sem CUDA
                response = input("CUDA não está disponível. O processamento será mais lento. Deseja continuar? (y/n): ")
                
                if response.lower() != 'y':
                    self.logger.info("Encerrando o programa por escolha do usuário...")
                    sys.exit(0)
                
                self.logger.info("Continuando sem aceleração CUDA")
                
        except ImportError:
            self.logger.warning("PyTorch não está instalado. Não foi possível verificar CUDA.")
            
            # Perguntar ao usuário se deseja continuar sem CUDA
            response = input("Não foi possível verificar CUDA. O processamento pode ser mais lento. Deseja continuar? (y/n): ")
            
            if response.lower() != 'y':
                self.logger.info("Encerrando o programa por escolha do usuário...")
                sys.exit(0)
            
            self.logger.info("Continuando sem verificação de CUDA")
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar CUDA: {e}")
            
            # Perguntar ao usuário se deseja continuar mesmo com erro
            response = input("Erro ao verificar CUDA. Deseja continuar mesmo assim? (y/n): ")
            
            if response.lower() != 'y':
                self.logger.info("Encerrando o programa por escolha do usuário...")
                sys.exit(0)
            
            self.logger.info("Continuando após erro na verificação de CUDA")
    
    def initialize_modules(self):
        """
        Inicializa todos os módulos necessários para o sistema.
        """
        self.logger.info("Inicializando módulos do sistema")
        
        try:
            # Aqui seriam importados e inicializados os demais módulos
            # De acordo com a estrutura do projeto
            # Por exemplo:
            # from modules.camera_connection import CameraManager
            # from modules.vision import YoloProcessor
            # etc.
            from modules.vision import YOLOProcessor, FoodVolumeEstimator
            
            # Informar sobre o estado do CUDA para os módulos que utilizam GPU
            # self.camera_manager = CameraManager(self.cameras_config)
            # self.vision_processor = YoloProcessor(use_cuda=self.cuda_available)
            # etc.
            
            self.logger.info("Todos os módulos inicializados com sucesso")
        except Exception as e:
            self.logger.exception(f"Falha ao inicializar módulos: {e}")
            sys.exit(1)
    
    def start(self):
        """
        Inicia o sistema de monitoramento.
        """
        if self.running:
            self.logger.warning("Sistema já está em execução")
            return
            
        self.logger.info("Iniciando sistema de monitoramento")
        self.running = True
        
        try:
            # Inicializar todos os módulos necessários
            self.initialize_modules()
            
            # Iniciar processamento principal em loop
            self.main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Interrupção de teclado detectada")
            self.stop()
        except Exception as e:
            self.logger.exception(f"Erro crítico durante execução: {e}")
            self.stop()
            sys.exit(1)
    
    def main_loop(self):
        """
        Loop principal de processamento do sistema.
        """
        self.logger.info("Iniciando loop principal de processamento")
        
        while self.running:
            try:
                # Aqui seria implementada a lógica principal de processamento
                # Por exemplo:
                # 1. Capturar imagens das câmeras
                # 2. Processar as imagens com modelo YOLO
                # 3. Analisar resultados e gerar métricas
                # 4. Enviar dados para dashboard
                # 5. Verificar comandos recebidos via API
                
                # Exemplo simplificado:
                # frames = self.camera_manager.get_frames()
                # results = self.vision_processor.process_frames(frames)
                # metrics = self.metrics_analyzer.analyze(results)
                # self.api_client.send_data(metrics)
                
                # Pausa para controle da taxa de processamento
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Erro durante processamento: {e}")
                # Decidir se deve continuar ou parar baseado na gravidade do erro
    
    def stop(self):
        """
        Para o sistema de monitoramento e realiza limpeza de recursos.
        """
        if not self.running:
            self.logger.warning("Sistema já está parado")
            return
            
        self.logger.info("Parando sistema de monitoramento")
        self.running = False
        
        # Aqui seriam implementadas rotinas de finalização de cada módulo
        # Por exemplo:
        # self.camera_manager.close()
        # self.api_client.close()
        
        self.logger.info("Sistema finalizado com sucesso")


def main():
    """
    Função principal para iniciar o sistema.
    """
    # Definir caminho de configuração (pode ser alterado via argumento de linha de comando)
    config_path = "config"
    
    # Criar e iniciar o sistema
    system = BuffetMonitoringSystem(config_path)
    system.start()


if __name__ == "__main__":
    main()
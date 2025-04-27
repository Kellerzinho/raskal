#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Teste com Vídeos - Sistema de Monitoramento de Buffet
Versão de testes que utiliza vídeos locais em vez de streams de câmeras ESP32Cam.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
import threading
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np


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


class VideoThread(threading.Thread):
    """
    Thread para gerenciar a leitura de um arquivo de vídeo específico.
    """
    
    def __init__(self, video_id, video_path, max_fps=15):
        """
        Inicializa a thread para um vídeo.
        
        Args:
            video_id: ID do vídeo (equivalente ao ID da câmera)
            video_path: Caminho para o arquivo de vídeo
            max_fps: Taxa máxima de quadros por segundo
        """
        super().__init__(name=f"VideoThread-{video_id}")
        self.logger = logging.getLogger(__name__)
        self.video_id = video_id
        self.video_path = video_path
        self.max_fps = max_fps
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def run(self):
        """
        Função principal da thread.
        """
        self.logger.info(f"Thread do vídeo {self.video_id} iniciada - Vídeo: {self.video_path}")
        self.running = True
        
        try:
            # Tentar abrir o vídeo
            if not Path(self.video_path).exists():
                self.logger.error(f"Arquivo de vídeo não encontrado: {self.video_path}")
                self.running = False
                return
                
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.logger.error(f"Não foi possível abrir o vídeo: {self.video_path}")
                self.running = False
                return
                
            self.logger.info(f"Vídeo {self.video_id} aberto com sucesso")
            
            # Calcular delay para limitar FPS
            frame_delay = 1.0 / self.max_fps
            
            # Loop para leitura dos frames
            last_frame_time = time.time()
            while self.running:
                # Controle de FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
                # Ler o próximo frame
                ret, frame = cap.read()
                last_frame_time = time.time()
                
                if not ret:
                    # Chegou ao fim do vídeo, reiniciar
                    self.logger.info(f"Fim do vídeo {self.video_id} alcançado, reiniciando...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Atualizar o frame atual
                with self.frame_lock:
                    self.current_frame = frame.copy()
            
            # Liberar recursos ao finalizar
            cap.release()
            self.logger.info(f"Thread do vídeo {self.video_id} finalizada")
            
        except Exception as e:
            self.logger.exception(f"Erro na thread do vídeo {self.video_id}: {e}")
            self.running = False
    
    def get_current_frame(self):
        """
        Obtém o frame atual capturado pela thread.
        
        Returns:
            numpy.ndarray: Frame atual ou None se não houver frame disponível
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def stop(self):
        """
        Para a execução da thread.
        """
        self.running = False


class BuffetMonitoringSystemTest:
    """
    Classe principal do sistema de teste de monitoramento de buffet.
    Usa vídeos locais em vez de câmeras ESP32Cam.
    """
    
    def __init__(self):
        """
        Inicializa o sistema de teste de monitoramento.
        """
        # Inicializar o gerenciador de logs primeiro
        self.logger_manager = LoggerManager()
        self.logger = logging.getLogger(__name__)
        
        self.running = False
        self.cuda_available = False
        self.vision_processor = None
        self.video_threads = []
        
        # Lista de vídeos para teste
        self.VIDEO_PATHS = [
            "videos/Local 2_cam2_20250416_214513.mp4",
            "videos/Local 4_cam4_20250416_214513.mp4",
            "videos/Local 5_cam5_20250416_214513.mp4",
            "videos/Local 6_cam6_20250416_214513.mp4",
            "videos/Local 10_cam10_20250416_214513.mp4",
            # Adicione mais caminhos de vídeos conforme necessário
        ]
        
        self.logger.info("Inicializando Sistema de Teste com Vídeos para Monitoramento de Buffet")
        
        # Verificar existência do diretório de vídeos
        videos_dir = Path("videos")
        if not videos_dir.exists():
            self.logger.warning(f"Diretório de vídeos não encontrado: {videos_dir}")
            videos_dir.mkdir(exist_ok=True)
            self.logger.info(f"Diretório de vídeos criado: {videos_dir}")
        
        # Verificar disponibilidade de CUDA
        self.check_cuda()
        
        self.logger.debug("Sistema de teste inicializado com sucesso")
    
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
    
    def start_video_threads(self):
        """
        Inicia uma thread para cada vídeo configurado.
        """
        self.logger.info(f"Iniciando {len(self.VIDEO_PATHS)} threads de vídeo")
        
        for i, video_path in enumerate(self.VIDEO_PATHS):
            video_path_obj = Path(video_path)
            
            if not video_path_obj.exists():
                self.logger.warning(f"Arquivo de vídeo não encontrado: {video_path}")
                continue
            
            # Extrair apenas o ID da câmera (camX) do nome do arquivo
            filename = video_path_obj.name
            parts = filename.split('_')
            if len(parts) >= 2:
                video_id = parts[1]  # Obter "camX"
            else:
                video_id = f"video{i+1}"
            
            # Criar e iniciar thread para este vídeo
            thread = VideoThread(video_id, video_path, max_fps=15)
            thread.daemon = True  # Threads daemon terminam quando o programa principal termina
            thread.start()
            
            # Adicionar à lista de threads
            self.video_threads.append(thread)
            
            self.logger.debug(f"Thread para vídeo {video_id} iniciada")
        
        self.logger.info("Todas as threads de vídeo foram iniciadas")
    
    def start(self):
        """
        Inicia o sistema de teste de monitoramento.
        """
        if self.running:
            self.logger.warning("Sistema já está em execução")
            return
            
        self.logger.info("Iniciando sistema de teste de monitoramento")
        self.running = True
        
        try:
            # Inicializar o processador YOLO
            self.logger.info("Inicializando processador de visão computacional")
            # from vision import YOLOProcessor
            # self.vision_processor = YOLOProcessor(
            #     model_path="models/FVBM.pt", 
            #     use_cuda=self.cuda_available,
            #     conf_threshold=0.5,
            #     iou_threshold=0.45
            # )
            
            # Iniciar threads para cada vídeo
            self.logger.info("Iniciando threads para cada vídeo")
            self.start_video_threads()
            
            # Loop principal
            self.logger.info("Sistema inicializado com sucesso. Pressione Ctrl+C para encerrar.")
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupção de teclado detectada")
            self.stop()
        except Exception as e:
            self.logger.exception(f"Erro crítico durante execução: {e}")
            self.stop()
            sys.exit(1)
    
    def stop(self):
        """
        Para o sistema de teste de monitoramento e realiza limpeza de recursos.
        """
        if not self.running:
            self.logger.warning("Sistema já está parado")
            return
            
        self.logger.info("Parando sistema de teste de monitoramento")
        self.running = False
        
        # Parar todas as threads de vídeo
        for thread in self.video_threads:
            if hasattr(thread, 'stop'):
                thread.stop()
        
        # Aguardar todas as threads terminarem (com timeout)
        for thread in self.video_threads:
            thread.join(timeout=2.0)
        
        # Limpar lista de threads
        self.video_threads.clear()
        
        self.logger.info("Sistema finalizado com sucesso")


def main():
    """
    Função principal para iniciar o sistema de teste.
    """
    # Criar e iniciar o sistema de teste
    system = BuffetMonitoringSystemTest()
    system.start()


if __name__ == "__main__":
    main()
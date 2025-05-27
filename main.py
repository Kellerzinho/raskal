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
import threading
import cv2
from datetime import datetime
from pathlib import Path
from api_server import APIServer
import dish_name_replacer



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



        """
        Para a execução da thread.
        """
        self.running = False


class CameraThread(threading.Thread):
    """
    Thread para gerenciar a conexão com uma câmera específica.
    Versão modificada que nunca encerra a thread, tentando continuamente reconectar.
    """
    
    def __init__(self, camera_id, camera_config, vision_processor=None):
        """
        Inicializa a thread para uma câmera.
        
        Args:
            camera_id: ID da câmera
            camera_config: Configuração da câmera
            vision_processor: Processador de visão computacional (YOLOProcessor)
        """
        super().__init__(name=f"CameraThread-{camera_id}")
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.running = False
        self.vision_processor = vision_processor
        self.frame_count = 0
        self.fps_limit = camera_config.get("max_fps", 15)
        self.frame_interval = 1.0 / self.fps_limit
        self.current_frame = None
        self.current_annotated_frame = None
        self.frame_lock = threading.Lock()
        self.detection_processor = None  # Será inicializado no método run
        self.frame_processor = None      # Será inicializado no método run
        
        # Parâmetros de reconexão
        self.reconnect_interval = 5  # segundos entre tentativas de reconexão
        self.connection_attempts = 0
        self.max_consecutive_failures = 5  # falhas consecutivas na leitura de frames
        self.consecutive_failures = 0
        
    def run(self):
        """
        Função principal da thread com loop de reconexão contínua.
        Esta função nunca encerra a thread enquanto running=True, 
        sempre retornando ao estágio de teste de conexão em caso de falha.
        """
        from netcam_connection import NetCamStudioConnection as CameraConnection

        self.logger.info(f"Thread da câmera {self.camera_id} iniciada")
        self.running = True
        
        # Inicializar processadores
        from processing import DetectionProcessor, FrameProcessor
        self.detection_processor = DetectionProcessor(data_file="buffet_data.json")
        self.frame_processor = FrameProcessor()
        
        # Loop principal - permanece ativo enquanto self.running for True
        while self.running:
            try:
                # ETAPA 1: Criar instância da câmera e testar conexão
                camera = CameraConnection(self.camera_config)
                self.connection_attempts += 1
            
                # ETAPA 2: Testar conexão com a stream
                self.logger.info(f"Estabelecendo conexão com a stream da câmera {self.camera_id}...")
                stream_success = camera.try_connect_to_stream(timeout=40)
                
                if not stream_success:
                    self.logger.warning(f"Falha ao conectar à stream da câmera {self.camera_id}. Tentando novamente em {self.reconnect_interval} segundos...")
                    time.sleep(self.reconnect_interval)
                    continue  # Voltar ao início do loop
                
                # ETAPA 3: Processar frames da câmera
                self.logger.info(f"Conexão com a stream da câmera {self.camera_id} estabelecida com sucesso")
                
                if self.vision_processor:
                    # Processar frames usando o modelo YOLO
                    self.process_frames(camera)
                else:
                    self.logger.warning(f"Processador de visão não disponível para a câmera {self.camera_id}")
                    time.sleep(self.reconnect_interval)
                
            except Exception as e:
                self.logger.error(f"Erro durante operação da câmera {self.camera_id}: {e}")
                time.sleep(self.reconnect_interval)
                
            # Se chegou aqui, é porque houve alguma falha no processamento
            # Voltamos ao início do loop principal para tentar reconectar
            self.logger.info(f"Reiniciando o ciclo de conexão para a câmera {self.camera_id}...")
        
        self.logger.info(f"Thread da câmera {self.camera_id} finalizada")
    
    def process_frames(self, camera):
        """
        Processa frames da câmera com detecção YOLO e contagem de falhas.
        Em caso de falhas recorrentes, retorna para permitir reconexão.
        
        Args:
            camera: Instância da conexão com a câmera
        """
        self.logger.info(f"Iniciando processamento de stream da câmera {self.camera_id} com modelo YOLO")
        
        # Resetar contador de falhas consecutivas
        self.consecutive_failures = 0
        
        try:
            # Abrir a stream usando OpenCV
            cap = cv2.VideoCapture(camera.stream_url)
            
            if not cap.isOpened():
                self.logger.error(f"Não foi possível abrir a stream da câmera {self.camera_id}")
                return  # Retornar ao loop principal para reconexão
            
            # Variáveis para controle de FPS
            last_frame_time = time.time()
            
            # Loop de processamento enquanto a câmera estiver conectada
            while self.running:
                # Controlar taxa de frames
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed < self.frame_interval:
                    # Esperar para respeitar o limite de FPS
                    time.sleep(self.frame_interval - elapsed)
                    continue
                    
                # Atualizar timestamp do último frame
                last_frame_time = time.time()
                
                # Ler o próximo frame
                ret, frame = cap.read()
                
                if not ret:
                    self.consecutive_failures += 1
                    self.logger.warning(f"Falha ao ler frame da câmera {self.camera_id} (falha #{self.consecutive_failures})")
                    
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.logger.error(f"Máximo de falhas consecutivas ({self.max_consecutive_failures}) atingido para câmera {self.camera_id}")
                        cap.release()
                        return  # Retornar ao loop principal para reconexão
                    
                    time.sleep(0.1)  # Pequena pausa antes de tentar ler o próximo frame
                    continue
                
                # Sucesso na leitura do frame - resetar contador de falhas
                self.consecutive_failures = 0
                
                # Incrementar contador de frames
                self.frame_count += 1
                
                # Salvar o frame atual
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Processar o frame com o modelo YOLO
                try:
                    # 1. Processar imagem com o modelo YOLO
                    results = self.vision_processor.model(frame)
                    
                    # Extrair detecções do resultado do modelo
                    detections = []
                    for detection in results[0].boxes.data:
                        x1, y1, x2, y2, confidence, class_id = detection
                        if confidence > self.vision_processor.conf_threshold:
                            class_id_int = int(class_id)
                            if class_id_int < len(results[0].names):
                                detections.append({
                                    'class': results[0].names[class_id_int],
                                    'confidence': float(confidence),
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                })
                    
                    # 2. Processar detecções (desenhar bounding boxes, calcular porcentagens)
                    annotated_frame, stats = self.detection_processor.process_detections(
                        self.camera_id, frame, detections, track_max_area=True
                    )
                    
                    # 3. Adicionar timestamp e informações visuais adicionais
                    annotated_frame = self.frame_processor.add_timestamp(annotated_frame, self.camera_id)
                    
                    # 4. Manter o frame processado para visualização
                    with self.frame_lock:
                        self.current_annotated_frame = annotated_frame
                    
                    # Log de detecções (limitado para não sobrecarregar o log)
                    if self.frame_count % 30 == 0 or len(detections) > 0:
                        self.logger.debug(f"Câmera {self.camera_id}: {len(detections)} detecções no frame {self.frame_count}")
                        
                        # Mostrar estatísticas das classes detectadas
                        if len(stats['classes']) > 0:
                            classes_str = ", ".join([f"{k}: {v}" for k, v in stats['classes'].items()])
                            self.logger.debug(f"Classes detectadas: {classes_str}")
                        
                        # Mostrar necessidades de reposição
                        if stats['needs_refill']:
                            self.logger.warning(f"Câmera {self.camera_id}: {len(stats['needs_refill'])} pratos precisam de reposição!")
                    
                except Exception as e:
                    self.logger.error(f"Erro ao processar frame da câmera {self.camera_id} com YOLO: {e}")
                    # Continuar tentando processar frames, não retornar ao loop principal
            
            # Liberar recursos ao sair do loop
            cap.release()
            self.logger.info(f"Processamento de frames da câmera {self.camera_id} finalizado")
            
        except Exception as e:
            self.logger.exception(f"Erro durante processamento da stream da câmera {self.camera_id}: {e}")
            # Deixar o erro propagar para o loop principal que irá reiniciar o ciclo
    
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
    
    def get_annotated_frame(self):
        """
        Obtém o frame atual com anotações (bounding boxes, etc.).
        
        Returns:
            numpy.ndarray: Frame com anotações ou None se não houver frame disponível
        """
        with self.frame_lock:
            if self.current_annotated_frame is not None:
                return self.current_annotated_frame.copy()
        return None
    
    def stop(self):
        """
        Para a execução da thread.
        """
        self.running = False


class BuffetMonitoringSystem:
    """
    Classe principal do sistema de monitoramento de buffet.
    Coordena todos os módulos e gerencia o ciclo de vida da aplicação.
    """
    
    def __init__(self, config_path="config", show_visualization=True):
        """
        Inicializa o sistema de monitoramento.
        
        Args:
            config_path: Caminho para o diretório de configurações
            show_visualization: Se True, mostra uma janela com os frames processados
        """
        # Inicializar o gerenciador de logs primeiro
        self.logger_manager = LoggerManager()
        self.logger = logging.getLogger(__name__)
        
        self.config_path = Path(config_path)
        self.running = False
        self.cuda_available = False
        self.vision_processor = None
        self.camera_threads = []
        self.show_visualization = show_visualization
        self.visualization_thread = None
        self.api_thread = None
        self.api_server = None

        # Configuração da API
        self.external_api_config = {
            'EXTERNAL_API_URL': "http://localhost:3300/api/replacements/updateReplacements",
            'SYNC_INTERVAL': 10,  # Intervalo em segundos
            'AUTH_TOKEN': None  # Token de autenticação (opcional)
        }

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
    
    def start_visualization_thread(self):
        """
        Inicia uma thread separada para visualização dos frames processados.
        """
        if not self.show_visualization:
            return
            
        self.visualization_thread = threading.Thread(
            target=self.visualization_loop,
            name="VisualizationThread"
        )
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        self.logger.info("Thread de visualização iniciada")
    
    def visualization_loop(self):
        """
        Loop principal da thread de visualização.
        Mostra os frames processados de todas as câmeras em uma janela.
        """
        try:
            from processing import FrameProcessor
            frame_processor = FrameProcessor()
            
            self.logger.info("Loop de visualização iniciado")
            
            while self.running:
                # Coletar frames processados de todas as threads ativas
                frames = []
                for thread in self.camera_threads:
                    if thread.is_alive():
                        frame = thread.get_annotated_frame()
                        if frame is not None:
                            # Redimensionar para um tamanho uniforme
                            frame = frame_processor.resize_frame(frame, target_width=480)
                            frames.append(frame)
                
                if frames:
                    # Combinar todos os frames em um grid
                    combined_frame = frame_processor.concat_frames(frames)
                    
                    if combined_frame is not None:
                        # Mostrar o frame combinado
                        cv2.imshow("Buffet Monitoring System", combined_frame)
                        
                        # Processar teclas (ESC para sair)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC
                            self.logger.info("Tecla ESC pressionada, encerrando visualização")
                            self.stop()
                            break
                
                # Pequeno delay para não sobrecarregar a CPU
                time.sleep(0.05)
                
            # Fechar todas as janelas ao sair
            cv2.destroyAllWindows()
            self.logger.info("Visualização encerrada")
            
        except Exception as e:
            self.logger.exception(f"Erro na thread de visualização: {e}")
    
    def start_camera_threads(self):
        """
        Inicia uma thread para cada câmera configurada.
        """
        self.logger.info(f"Iniciando {len(self.cameras_config['cameras'])} threads de câmera")
        
        for camera_config in self.cameras_config["cameras"]:
            camera_id = camera_config["id"]
            
            # Criar e iniciar thread para esta câmera
            thread = CameraThread(camera_id, camera_config, self.vision_processor)
            thread.daemon = True  # Threads daemon terminam quando o programa principal termina
            thread.start()
            
            # Adicionar à lista de threads
            self.camera_threads.append(thread)
            
            self.logger.debug(f"Thread para câmera {camera_id} iniciada")
        
        self.logger.info("Todas as threads de câmera foram iniciadas")
    
    def show_statistics(self):
        """
        Mostra estatísticas do sistema e carrega os dados mais recentes do arquivo JSON.
        """
        try:
            from processing import DetectionProcessor
            
            # Tentar carregar os dados do arquivo JSON
            processor = DetectionProcessor(data_file="buffet_data.json")
            buffet_data = processor.load_area_data()
            
            # Estatísticas básicas
            active_threads = sum(1 for t in self.camera_threads if t.is_alive())
            
            self.logger.info(f"=== Estatísticas do Sistema ===")
            self.logger.info(f"Threads ativas: {active_threads}/{len(self.camera_threads)}")
            
            # Mostrar dados do arquivo JSON
            if buffet_data:
                camera_count = len(buffet_data["address"])
                total_objects = sum(len(pratos) for pratos in buffet_data["address"].values())
                
                self.logger.info(f"Dados carregados de {camera_count} câmeras, {total_objects} pratos monitorados")
                
                # Listar pratos com baixa porcentagem (que precisam de reposição)
                needs_refill = []
                for camera_id, pratos in buffet_data.items():
                    for prato_name, data in pratos.items():
                        if data['percentage'] < 0.3:
                            needs_refill.append({
                                'camera': camera_id,
                                'nome': prato_name,
                                'percentage': data['percentage']
                            })
                
                if needs_refill:
                    self.logger.warning(f"{len(needs_refill)} pratos precisam de reposição!")
                    for item in needs_refill[:3]:  # Mostrar apenas os 3 primeiros para não sobrecarregar o log
                        self.logger.warning(f"  - {item['camera']}: {item['nome']} ({item['percentage']*100:.1f}%)")
                    if len(needs_refill) > 3:
                        self.logger.warning(f"  ... e mais {len(needs_refill)-3} pratos")
            
            self.logger.info(f"===============================")
            
        except Exception as e:
            self.logger.error(f"Erro ao mostrar estatísticas: {e}")
    
    def start_api_thread(self):
        """
        Inicia a thread de sincronização com a API externa.
        """
        self.logger.info("Iniciando thread de sincronização com API externa")
        
        # Importar a classe APIThread
        from api_thread import APIThread
        
        # Criar e iniciar a thread
        self.api_thread = APIThread(
            api_url=self.external_api_config['EXTERNAL_API_URL'],
            sync_interval=self.external_api_config['SYNC_INTERVAL'],
            token=self.external_api_config['AUTH_TOKEN']
        )
        self.api_thread.daemon = True
        self.api_thread.start()
        
        self.logger.debug("Thread de API iniciada com sucesso")

    def start_api_server(self):
        """
        Inicia o servidor API para receber solicitações do dashboard.
        """
        self.logger.info("Iniciando servidor API na porta 3320")
        
        # Criar e iniciar o servidor API
        self.api_server = APIServer(
            port=3320,
            data_file="buffet_data.json",
            system_instance=self  # Passa a instância atual do sistema para permitir encerramento
        )
        self.api_server.start()
        
        self.logger.debug("Servidor API iniciado com sucesso")

    def start(self):
        """
        Inicia o sistema de monitoramento.
        """

        try:
            data_file = "buffet_data.json"
            if os.path.exists(data_file):
                os.remove(data_file)
                self.logger.info(f"Arquivo {data_file} removido com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao remover arquivo buffet_data.json: {e}")

        if self.running:
            self.logger.warning("Sistema já está em execução")
            return
            
        self.logger.info("Iniciando sistema de monitoramento")
        self.running = True
        
        try:
            # Inicializar o processador YOLO
            self.logger.info("Inicializando processador de visão computacional")
            from vision import YOLOProcessor
            self.vision_processor = YOLOProcessor(
                model_path="models/FVBM.pt", 
                use_cuda=self.cuda_available,
                conf_threshold=0.5,
                iou_threshold=0.45
            )
            
            # Iniciar threads para cada câmera
            self.logger.info("Iniciando threads para cada câmera")
            self.start_camera_threads()

            #Iniciar thread para comunicação com o dashboard
            self.start_api_thread()

            #Iniciar servidor para receber requests do dashboard
            self.start_api_server()
            
            # Iniciar thread de visualização se necessário
            if self.show_visualization:
                self.start_visualization_thread()
            
            # Loop principal
            self.logger.info("Sistema inicializado com sucesso. Pressione Ctrl+C para encerrar.")
            
            # Loop principal com exibição de estatísticas periódicas
            stats_interval = 10  # segundos
            last_stats_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Mostrar estatísticas a cada intervalo definido
                if current_time - last_stats_time >= stats_interval:
                    self.show_statistics()
                    last_stats_time = current_time
                
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
        Para o sistema de monitoramento e realiza limpeza de recursos.
        """
        if not self.running:
            self.logger.warning("Sistema já está parado")
            return
            
        self.logger.info("Parando sistema de monitoramento")
        self.running = False
        
        # Parar todas as threads de câmera
        for thread in self.camera_threads:
            if hasattr(thread, 'stop'):
                thread.stop()
        
        # Parar a thread de API se estiver ativa
        if self.api_thread and hasattr(self.api_thread, 'stop'):
            self.api_thread.stop()

        # Aguardar a thread de API terminar
        if self.api_thread:
            self.api_thread.join(timeout=2.0)

        # Parar o servidor API se estiver ativo
        if self.api_server:
            self.api_server.stop()

        # Aguardar todas as threads terminarem (com timeout)
        for thread in self.camera_threads:
            thread.join(timeout=2.0)
        
        # Limpar lista de threads
        self.camera_threads.clear()
        
        try:
            data_file = "buffet_data.json"
            if os.path.exists(data_file):
                os.remove(data_file)
                self.logger.info(f"Arquivo {data_file} removido com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao remover arquivo buffet_data.json: {e}")

        # Fechar janelas OpenCV
        cv2.destroyAllWindows()
        
        self.logger.info("Sistema finalizado com sucesso")


def main():
    """
    Função principal para iniciar o sistema.
    """
    import argparse
    
    # Parse de argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Sistema de Monitoramento de Buffet")
    parser.add_argument("--no-display", action="store_true", help="Desativa a visualização dos frames")
    parser.add_argument("--config-path", default="config", help="Caminho para o diretório de configuração")
    args = parser.parse_args()
    
    # Criar e iniciar o sistema
    system = BuffetMonitoringSystem(
        config_path=args.config_path,
        show_visualization=not args.no_display
    )
    system.start()


if __name__ == "__main__":
    main()
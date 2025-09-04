import json
import logging
import threading
import time
from pathlib import Path
import cv2
from datetime import datetime, timedelta

# Importações refatoradas
from api_server import APIServer
from camera_worker import CameraThread
from vision_model import YOLOProcessor
from utils.dish_name_mapper import DishNameReplacer
from detection_processor import FrameProcessor, DetectionProcessor
from utils.status_monitor import StatusMonitor

class BuffetMonitoringSystem:
    """
    Classe principal que orquestra todo o sistema de monitoramento do buffet.
    """
    
    def __init__(self, config_path="config", show_visualization=True):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_path)
        self.show_visualization = show_visualization
        self.running = False
        self.cameras_config = None
        self.camera_threads = {}
        self.last_displayed_frame_time = {}
        
        self.frame_processor = FrameProcessor()
        self.dish_name_replacer = None
        self.status_monitor = None
        
        self.logger.info("Carregando configurações...")
        self.load_configs()
        self.initialize_components() # Novo método para inicializar componentes
        
        self.running = True
        
        # O try/finally garante que o método stop seja chamado ao sair
        try:
            self.start_camera_threads()
            self.start_api_server()
            
            # Inicia o monitor de status *depois* que tudo estiver pronto
            if self.status_monitor:
                self.status_monitor.start()

            if self.show_visualization:
                self.logger.info("Visualização ativada. Pressione 'q' na janela para sair.")
                
                # Dimensões para redimensionamento
                target_height = 360  # Altura padrão para cada frame na grade

                while self.running:
                    frames_to_show = []
                    # Coleta os frames mais recentes de todas as threads
                    for cam_id, thread in self.camera_threads.items():
                        frame = thread.get_annotated_frame()
                        # A verificação de None foi removida, pois a thread agora sempre tem um frame
                        
                        # Redimensiona o frame para uma altura padrão, mantendo a proporção
                        h, w, _ = frame.shape
                        scale = target_height / h
                        target_width = int(w * scale)
                        resized_frame = cv2.resize(frame, (target_width, target_height))
                        frames_to_show.append(resized_frame)
                    
                    if frames_to_show:
                        # Concatena os frames em uma única imagem
                        combined_frame = self.frame_processor.concat_frames(frames_to_show)
                        if combined_frame is not None and combined_frame.size > 0:
                            cv2.imshow("Buffet Monitor", combined_frame)
                    
                    # O waitKey é crucial para que o OpenCV processe os eventos da janela
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Tecla 'q' pressionada. Encerrando...")
                        self.stop()
                        break
                    
                    # Pequena pausa para não sobrecarregar a CPU
                    time.sleep(0.02)
            else:
                # Modo headless
                while self.running:
                    time.sleep(1)
        finally:
            if self.running:
                self.logger.info("Encerrando o sistema a partir do bloco finally...")
                self.stop()

    def load_configs(self):
        """
        Carrega as configurações de câmeras e nomes de pratos.
        """
        try:
            # Carrega configuração das câmeras
            cameras_config_file = self.config_dir / "cameras.json"
            with open(cameras_config_file, 'r') as f:
                self.cameras_config = json.load(f)
            
            # Criar mapa de informações de câmera para fácil acesso
            self.camera_info_map = {
                cam["id"]: cam
                for cam in self.cameras_config.get("cameras", [])
            }
            
            # Carrega configuração do dashboard
            dashboard_config_file = self.config_dir / "dashboard.json"
            if dashboard_config_file.exists():
                with open(dashboard_config_file, 'r') as f:
                    self.dashboard_config = json.load(f)

            # Carrega mapeamento de nomes de pratos
            names_config_file = self.config_dir / "nomes.json"
            if names_config_file.exists():
                self.dish_name_replacer = DishNameReplacer(names_config_file)
            else:
                self.logger.warning("Arquivo de mapeamento de nomes não encontrado")
            
            self.logger.info("Configurações carregadas com sucesso")
        except FileNotFoundError as e:
            self.logger.error(f"Arquivo de configuração não encontrado: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar JSON de configuração: {e}")
            raise

    def initialize_components(self):
        """
        Inicializa os componentes principais como o processador de visão e de detecção.
        """
        self.logger.info("Inicializando componentes do sistema...")
        
        # Inicializa o processador de visão (YOLO)
        try:
            self.vision_processor = YOLOProcessor(model_path="models/FVBM.pt", retina_masks=True)
            self.logger.info("Processador de visão (YOLO) inicializado.")
        except Exception as e:
            self.logger.error(f"Falha ao inicializar o processador de visão: {e}", exc_info=True)
            # Continua sem visão, mas a visualização será desativada se estiver ativa
            if self.show_visualization:
                self.logger.warning("Visualização será desativada devido à falha na inicialização da visão.")
                self.show_visualization = False

        # Inicializa o processador de detecção centralizado
        dashboard_url = self.dashboard_config.get("url") if self.dashboard_config else None
        auth_token = self.dashboard_config.get("auth_token") if self.dashboard_config else None
        
        self.detection_processor = DetectionProcessor(
            camera_info=self.camera_info_map,
            data_file="config/buffet_data.json",  # Caminho corrigido
            dish_name_replacer=self.dish_name_replacer,
            dashboard_url=dashboard_url,
            auth_token=auth_token
        )
        self.logger.info("Processador de detecção centralizado inicializado.")

        # Inicializar o monitor de status com os IDs das câmeras
        camera_ids = [cam["id"] for cam in self.cameras_config.get("cameras", [])]
        self.status_monitor = StatusMonitor(camera_ids)

        self.stop_event = threading.Event()

        # Adiciona o scheduler para limpeza diária
        self.cleanup_thread = threading.Thread(target=self._daily_cleanup_scheduler, daemon=True)

    def _daily_cleanup_scheduler(self):
        """
        Verifica a cada minuto se é meia-noite para executar a rotina de limpeza.
        """
        self.logger.info("Agendador de limpeza diária iniciado.")
        while not self.stop_event.is_set():
            # Calcula a próxima meia-noite
            now = datetime.now()
            next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Checa se a hora atual está próxima da meia-noite (margem de 1 minuto para garantir execução)
            if now.hour == 0 and now.minute == 0:
                self.logger.info("Meia-noite alcançada. Executando limpeza de dados diários.")
                try:
                    self.detection_processor.reset_daily_data()
                    self.logger.info("Limpeza diária concluída com sucesso.")
                    # Dorme por um pouco mais de um minuto para não rodar de novo no mesmo minuto
                    time.sleep(61)
                except Exception as e:
                    self.logger.error(f"Ocorreu um erro durante a execução da limpeza diária: {e}")
            
            # Dorme por 60 segundos antes de checar novamente
            time.sleep(60)
        self.logger.info("Agendador de limpeza diária finalizado.")

    def check_cuda(self):
        """
        Verifica a disponibilidade do CUDA e inicializa o processador de visão.
        """
        self.logger.info("Verificando disponibilidade do CUDA...")
        try:
            # Esta é uma maneira simplificada. A lógica real pode estar dentro de YOLOProcessor.
            # Supondo que YOLOProcessor lida com a seleção de dispositivo (CPU/GPU).
            self.vision_processor = YOLOProcessor(model_path="models/FVBM.pt", retina_masks=True)
            # A lógica original para verificar torch e cuda foi abstraída para dentro do YOLOProcessor
            self.logger.info("Processador de visão (YOLO) inicializado.")
        except Exception as e:
            self.logger.error(f"Falha ao inicializar o processador de visão: {e}")
            self.vision_processor = None
            # O sistema pode continuar sem visão, mas registrará avisos.
            if self.show_visualization:
                self.logger.warning("Visualização será desativada devido à falha na inicialização da visão.")
                self.show_visualization = False

    def start_visualization_thread(self):
        """
        Inicia uma thread para a visualização das câmeras.
        """
        if self.show_visualization:
            self.logger.info("Iniciando thread de visualização")
            vis_thread = threading.Thread(target=self.visualization_loop, name="VisualizationThread")
            vis_thread.daemon = True
            vis_thread.start()

    def visualization_loop(self):
        """
        Loop que exibe os frames anotados das câmeras.
        """
        self.logger.info("Visualização ativada. Pressione 'q' na janela para sair.")
        try:
            while self.running:
                frames_to_show = []
                for cam_id, thread in self.camera_threads.items():
                    frame = thread.get_annotated_frame()
                    if frame is not None:
                        frames_to_show.append(frame)
                
                if frames_to_show:
                    combined_frame = self.frame_processor.concat_frames(frames_to_show)
                    cv2.imshow("Buffet Monitor", combined_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Tecla 'q' pressionada. Encerrando o sistema.")
                    self.stop()
                    break
                time.sleep(0.05) # Pequena pausa para não sobrecarregar a CPU
        except Exception as e:
            self.logger.error(f"Erro na thread de visualização: {e}", exc_info=True)
        finally:
            cv2.destroyAllWindows()

    def start_camera_threads(self):
        """
        Inicia uma thread para cada câmera configurada, passando o processador de detecção.
        """
        self.logger.info("Iniciando threads das câmeras...")
        self.cleanup_thread.start()
        for camera_config in self.cameras_config.get("cameras", []):
            camera_id = camera_config["id"]
            thread = CameraThread(
                camera_id, 
                camera_config, 
                self.vision_processor, 
                self.detection_processor,
                self.frame_processor,
                self.status_monitor
            )
            self.camera_threads[camera_id] = thread
            thread.start()
        self.logger.info(f"{len(self.camera_threads)} threads de câmera iniciadas.")

    def start_api_server(self):
        """
        Inicia o servidor de API para consulta externa.
        """
        self.logger.info("Iniciando servidor da API...")
        self.api_server = APIServer(
            camera_threads=self.camera_threads, 
            camera_info=self.camera_info_map
        )
        api_thread = threading.Thread(target=self.api_server.run, name="APIServerThread")
        api_thread.daemon = True
        api_thread.start()

    def start(self):
        """
        Inicia os componentes do sistema.
        """
        try:
            self.load_configs()
            self.initialize_components() # Novo método para inicializar componentes
            
            self.running = True
            
            self.start_camera_threads()
            self.start_api_server()
            
            if self.show_visualization:
                self.logger.info("Visualização ativada. Pressione 'q' na janela para sair.")
                
                # Dimensões para redimensionamento
                target_height = 360  # Altura padrão para cada frame na grade

                while self.running:
                    frames_to_show = []
                    # Coleta os frames mais recentes de todas as threads
                    for cam_id, thread in self.camera_threads.items():
                        frame = thread.get_annotated_frame()
                        # A verificação de None foi removida, pois a thread agora sempre tem um frame
                        
                        # Redimensiona o frame para uma altura padrão, mantendo a proporção
                        h, w, _ = frame.shape
                        scale = target_height / h
                        target_width = int(w * scale)
                        resized_frame = cv2.resize(frame, (target_width, target_height))
                        frames_to_show.append(resized_frame)
                    
                    if frames_to_show:
                        # Concatena os frames em uma única imagem
                        combined_frame = self.frame_processor.concat_frames(frames_to_show)
                        if combined_frame is not None and combined_frame.size > 0:
                            cv2.imshow("Buffet Monitor", combined_frame)
                    
                    # O waitKey é crucial para que o OpenCV processe os eventos da janela
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Tecla 'q' pressionada. Encerrando...")
                        self.stop()
                        break
                    
                    # Pequena pausa para não sobrecarregar a CPU
                    time.sleep(0.02)
            else:
                # Modo headless
                while self.running:
                    time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Interrupção de teclado recebida. Encerrando...")
        except Exception as e:
            self.logger.exception(f"Erro fatal no sistema: {e}")
        finally:
            self.stop()

    def stop(self):
        """
        Para todas as threads e encerra o sistema de forma limpa.
        """
        if not self.running:
            return  # Evita chamadas múltiplas
        
        self.logger.info("Iniciando o processo de encerramento do sistema...")
        self.running = False
        
        self.stop_event.set()
        
        if hasattr(self, 'api_server') and self.api_server:
            self.api_server.stop()

        if self.status_monitor:
            self.status_monitor.stop()

        self.logger.info("Aguardando finalização das threads das câmeras...")
        for thread in self.camera_threads.values():
            thread.stop()
            thread.join()
        
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
            self.logger.info("Aguardando finalização da thread de limpeza...")
            self.cleanup_thread.join(timeout=5)

        self.logger.info("Todas as threads foram finalizadas.")

        # Fechar janelas de visualização
        cv2.destroyAllWindows()

        self.logger.info("Sistema encerrado.") 
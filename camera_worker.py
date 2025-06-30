import logging
import threading
import time
import cv2

from camera_connection import NetCamStudioConnection as CameraConnection
from detection_processor import DetectionProcessor, FrameProcessor

class CameraThread(threading.Thread):
    """
    Thread para gerenciar a conexão com uma câmera específica.
    Versão modificada que nunca encerra a thread, tentando continuamente reconectar.
    """
    
    def __init__(self, camera_id, camera_config, vision_processor=None, camera_info_map=None, dish_name_replacer=None, frame_processor=None):
        """
        Inicializa a thread para uma câmera.
        
        Args:
            camera_id: ID da câmera
            camera_config: Configuração da câmera
            vision_processor: Processador de visão computacional (YOLOProcessor)
            camera_info_map: Mapa com informações de todas as câmeras
            dish_name_replacer: Instância para traduzir nomes de pratos
            frame_processor: Instância do processador de frames para anotações
        """
        super().__init__(name=f"CameraThread-{camera_id}")
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.running = False
        self.vision_processor = vision_processor
        self.camera_info_map = camera_info_map
        self.dish_name_replacer = dish_name_replacer
        self.frame_processor = frame_processor
        self.frame_count = 0
        self.fps_limit = camera_config.get("max_fps", 15)
        self.frame_interval = 1.0 / self.fps_limit
        self.current_frame = None
        self.current_annotated_frame = None
        self.frame_lock = threading.Lock()
        self.detection_processor = None
        
        self.reconnect_interval = 5
        self.connection_attempts = 0
        self.max_consecutive_failures = 5
        self.consecutive_failures = 0
        
    def run(self):
        """
        Função principal da thread com loop de reconexão contínua.
        """
        self.logger.info(f"Thread da câmera {self.camera_id} iniciada")
        self.running = True
        
        self.detection_processor = DetectionProcessor(
            camera_info=self.camera_info_map,
            data_file="buffet_data.json",
            dish_name_replacer=self.dish_name_replacer
        )
        
        while self.running:
            try:
                camera = CameraConnection(self.camera_config)
                self.connection_attempts += 1
            
                self.logger.info(f"Estabelecendo conexão com a stream da câmera {self.camera_id}...")
                stream_success = camera.try_connect_to_stream(timeout=40)
                
                if not stream_success:
                    self.logger.warning(f"Falha ao conectar à stream da câmera {self.camera_id}. Tentando novamente em {self.reconnect_interval} segundos...")
                    time.sleep(self.reconnect_interval)
                    continue
                
                self.logger.info(f"Conexão com a stream da câmera {self.camera_id} estabelecida com sucesso")
                
                if self.vision_processor:
                    self.process_frames(camera)
                else:
                    self.logger.warning(f"Processador de visão não disponível para a câmera {self.camera_id}")
                    time.sleep(self.reconnect_interval)
                
            except Exception as e:
                self.logger.error(f"Erro durante operação da câmera {self.camera_id}: {e}")
                time.sleep(self.reconnect_interval)
                
            self.logger.info(f"Reiniciando o ciclo de conexão para a câmera {self.camera_id}...")
        
        self.logger.info(f"Thread da câmera {self.camera_id} finalizada")
    
    def process_frames(self, camera):
        """
        Processa frames da câmera com detecção YOLO.
        Este método agora contém seu próprio loop de reconexão de stream
        para ser mais resiliente a falhas temporárias.
        """
        self.logger.info(f"Iniciando processamento de stream da câmera {self.camera_id} com modelo YOLO")
        cap = None
        
        while self.running:
            try:
                # Se o objeto de captura não existe ou não está aberto, tente (re)abrir.
                if cap is None or not cap.isOpened():
                    self.logger.info(f"Abrindo stream de vídeo para a câmera {self.camera_id}...")
                    cap = cv2.VideoCapture(camera.stream_url)
                    if not cap.isOpened():
                        self.logger.warning(f"Não foi possível abrir a stream para {self.camera_id}. Tentando novamente em {self.reconnect_interval}s...")
                        time.sleep(self.reconnect_interval)
                        continue
                
                # Controle de FPS
                last_frame_time = time.time()

                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Falha ao ler frame da câmera {self.camera_id}. Tentando reabrir a stream...")
                    cap.release()
                    cap = None  # Força a reabertura na próxima iteração
                    time.sleep(1)
                    continue

                # Lógica de processamento de frame
                annotated_frame = frame.copy()
                results = self.vision_processor.process_frame(frame)
                
                if self.detection_processor and results and len(results[0].boxes) > 0:
                    annotated_frame, _ = self.detection_processor.process_detections(
                        annotated_frame,
                        self.camera_id,
                        results[0].boxes,
                        self.vision_processor.model.names
                    )
                
                if self.frame_processor:
                    self.frame_processor.add_timestamp(annotated_frame, self.camera_id)
                
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_annotated_frame = annotated_frame

                # Garante o limite de FPS
                elapsed = time.time() - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)

            except Exception as e:
                self.logger.error(f"Erro no loop de processamento da câmera {self.camera_id}: {e}", exc_info=True)
                if cap:
                    cap.release()
                cap = None
                time.sleep(self.reconnect_interval)
        
        # Garante que o recurso seja liberado ao sair
        if cap:
            cap.release()
        self.logger.warning(f"Processamento de frames da câmera {self.camera_id} interrompido.")

    def get_current_frame(self):
        """
        Retorna o frame atual (sem anotações).
        """
        with self.frame_lock:
            return self.current_frame

    def get_annotated_frame(self):
        """
        Retorna o frame atual com anotações.
        """
        with self.frame_lock:
            return self.current_annotated_frame

    def stop(self):
        """
        Para a execução da thread.
        """
        self.running = False 
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
    
    def __init__(self, camera_id, camera_config, vision_processor, detection_processor, frame_processor):
        """
        Inicializa a thread para uma câmera.
        
        Args:
            camera_id: ID da câmera
            camera_config: Configuração da câmera
            vision_processor: Processador de visão computacional (YOLOProcessor)
            detection_processor: Processador de detecções para salvar dados (instância centralizada)
            frame_processor: Instância do processador de frames para anotações
        """
        super().__init__(name=f"CameraThread-{camera_id}")
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.running = False
        self.vision_processor = vision_processor
        self.detection_processor = detection_processor # Recebe a instância
        self.frame_processor = frame_processor
        self.frame_count = 0
        self.fps_limit = camera_config.get("max_fps", 15)
        self.frame_interval = 1.0 / self.fps_limit
        self.current_frame = None
        self.current_annotated_frame = None
        self.frame_lock = threading.Lock()
        
        self.reconnect_interval = 5
        self.connection_attempts = 0
        self.max_consecutive_failures = 5
        self.consecutive_failures = 0
        
        self.last_boxes = None  # Armazena as últimas caixas de detecção
        self.last_model_names = None 
        self.last_percentages = {}  # Armazena as últimas porcentagens por nome de prato
        
        # Inicializa com um frame de status "Aguardando"
        width = self.camera_config.get("width", 800)
        height = self.camera_config.get("height", 600)
        self.current_annotated_frame = self.frame_processor.create_status_frame(f"Aguardando {self.camera_id}...", width, height)
        
    def run(self):
        """
        Função principal da thread com loop de reconexão contínua.
        """
        self.logger.debug(f"Thread da câmera {self.camera_id} iniciada")
        self.running = True
        
        # A inicialização do detection_processor foi movida para o BuffetMonitoringSystem
        
        while self.running:
            try:
                camera = CameraConnection(self.camera_config)
            
                self.logger.debug(f"Estabelecendo conexão com a stream da câmera {self.camera_id}...")
                
                # Gera um frame de status enquanto conecta
                width = self.camera_config.get("width", 800)
                height = self.camera_config.get("height", 600)
                placeholder = self.frame_processor.create_status_frame(f"Conectando a {self.camera_id}...", width, height)
                with self.frame_lock:
                    self.current_annotated_frame = placeholder

                stream_success = camera.try_connect_to_stream(timeout=40)
                
                if not stream_success:
                    self.logger.warning(f"Falha ao conectar à stream da câmera {self.camera_id}. Tentando novamente em {self.reconnect_interval} segundos...")
                    
                    # Gera um frame de status de falha
                    placeholder = self.frame_processor.create_status_frame(f"Falha - {self.camera_id}", width, height)
                    with self.frame_lock:
                        self.current_annotated_frame = placeholder
                        
                    time.sleep(self.reconnect_interval)
                    continue
                
                self.logger.debug(f"Conexão com a stream da câmera {self.camera_id} estabelecida com sucesso")
                
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
        Este método agora retorna ao loop principal de conexão em caso de falhas
        persistentes para garantir uma reconexão mais robusta.
        """
        self.logger.debug(f"Iniciando processamento de stream da câmera {self.camera_id} com modelo YOLO")
        
        self.logger.debug(f"Abrindo stream de vídeo para a câmera {self.camera_id}...")
        cap = cv2.VideoCapture(camera.stream_url)
        
        if not cap.isOpened():
            self.logger.warning(f"Não foi possível abrir a stream para {self.camera_id}. A conexão será tentada novamente.")
            return  # Retorna para o loop principal em run()

        self.consecutive_failures = 0  # Reseta o contador ao iniciar um novo processamento

        while self.running:
            try:
                # Controle de FPS
                last_frame_time = time.time()

                ret, frame = cap.read()

                if not ret:
                    self.consecutive_failures += 1
                    self.logger.warning(f"Falha ao ler frame da câmera {self.camera_id} ({self.consecutive_failures}/{self.max_consecutive_failures}).")
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.logger.error(f"Atingido o limite de falhas consecutivas para {self.camera_id}. Reiniciando conexão completa.")
                        break  # Sai do loop de processamento para forçar reconexão no método run()
                    
                    time.sleep(1)  # Pausa para evitar um loop muito rápido em caso de falha
                    continue

                # Sucesso na leitura, reseta o contador
                self.consecutive_failures = 0

                # Processar com YOLO
                results = self.vision_processor.process_frame(frame)
                
                # Verificar se há detecções
                if self.detection_processor and results and len(results[0].boxes) > 0:
                    # Se houver detecções, processá-las e armazená-las para persistência
                    self.last_boxes = results[0].boxes
                    self.last_model_names = self.vision_processor.model.names
                    
                    annotated_frame, stats = self.detection_processor.process_detections(
                        frame.copy(),
                        self.camera_id,
                        self.last_boxes,
                        self.last_model_names
                    )
                    
                    # Capturar as porcentagens calculadas para usar nas caixas persistentes
                    self.last_percentages = {}
                    for key, percentage in stats.get('area_percentages', {}).items():
                        # key tem formato "camera_id_dish_name", extrair apenas o dish_name
                        if key.startswith(f"{self.camera_id}_"):
                            dish_name = key[len(f"{self.camera_id}_"):]
                            self.last_percentages[dish_name] = percentage
                else:
                    # Se não houver detecções, usar as últimas caixas conhecidas
                    if self.last_boxes is not None:
                        annotated_frame = self.detection_processor.draw_persistent_boxes(
                            frame.copy(),
                            self.last_boxes,
                            self.last_model_names,
                            self.camera_id,
                            self.last_percentages
                        )
                    else:
                        # Se não houver detecções nem caixas antigas, o frame é uma cópia
                        annotated_frame = frame.copy()
                
                # Adicionar timestamp ao frame final (com ou sem detecções)
                if self.frame_processor:
                    self.frame_processor.add_timestamp(annotated_frame, self.camera_id)
                
                # Atualizar os frames na thread
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_annotated_frame = annotated_frame

                # Garante o limite de FPS
                elapsed = time.time() - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)

            except Exception as e:
                self.logger.error(f"Erro no loop de processamento da câmera {self.camera_id}: {e}", exc_info=True)
                break  # Sai do loop em caso de exceção para forçar reconexão

        # Garante que o recurso seja liberado ao sair
        if cap:
            cap.release()
        self.logger.warning(f"Processamento de frames da câmera {self.camera_id} interrompido. Retornando ao gerenciador de conexão.")

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
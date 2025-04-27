#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Visão Computacional - Sistema de Monitoramento de Buffet
Responsável por implementar e configurar o modelo YOLOv11x pré-treinado
para detecção de objetos em imagens de câmeras de buffet.
"""

import os
import logging
import numpy as np
from pathlib import Path
import cv2
import torch


class YOLOProcessor:
    """
    Classe responsável por carregar e gerenciar o modelo YOLOv11x para detecção de objetos.
    Processa imagens das câmeras e retorna resultados de detecção.
    """
    
    def __init__(self, model_path="models/FVBM.pt", use_cuda=False, conf_threshold=0.5, iou_threshold=0.45):
        """
        Inicializa o processador YOLO.
        
        Args:
            model_path: Caminho para o arquivo do modelo pré-treinado
            use_cuda: Se True, utiliza GPU para processamento
            conf_threshold: Limiar de confiança para detecções
            iou_threshold: Limiar de IOU para supressão de não-máximos
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.use_cuda = use_cuda
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = None
        self.model = None
        self.class_names = []
        
        # Inicializar o modelo
        self._load_model()
        
    def _load_model(self):
        """
        Carrega o modelo YOLOv11x pré-treinado e configura o dispositivo.
        """
        self.logger.info(f"Carregando modelo YOLOv11x de {self.model_path}")
        
        try:
            # Verificar se o arquivo do modelo existe
            if not self.model_path.exists():
                self.logger.error(f"Arquivo do modelo não encontrado: {self.model_path}")
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            # Configurar dispositivo
            if self.use_cuda and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.logger.info(f"Utilizando GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Utilizando CPU para inferência")
            
            # Carregar modelo
            try:
                # Tenta carregar usando o método nativo do PyTorch
                self.model = torch.load(self.model_path, map_location=self.device)
                
                # Tentar extrair modelo do dicionário state_dict, se aplicável
                if isinstance(self.model, dict) and 'model' in self.model:
                    self.model = self.model['model']
                
                # Verificar se o modelo está no formato correto, se não, tentar via hub
                if not hasattr(self.model, 'forward'):
                    self.logger.info("Modelo não é um módulo PyTorch, tentando carregar via YOLO")
                    # Tentar via YOLO do Ultralytics
                    try:
                        from ultralytics import YOLO
                        self.model = YOLO(self.model_path)
                        self.logger.info("Modelo carregado via Ultralytics YOLO")
                    except ImportError:
                        self.logger.error("Ultralytics não está instalado. Não foi possível carregar o modelo.")
                        raise
                
                self.logger.info("Modelo YOLOv11x carregado com sucesso")
                
                # Carregar nomes das classes se disponíveis
                self._load_class_names()
                
            except Exception as e:
                self.logger.error(f"Erro ao carregar modelo: {e}")
                raise
                
        except Exception as e:
            self.logger.exception(f"Falha ao inicializar o modelo YOLOv11x: {e}")
            raise
    
    def _load_class_names(self):
        """
        Carrega nomes das classes a partir de um arquivo ou do modelo.
        """
        try:
            # Primeiro, verificar se há um arquivo de classes no mesmo diretório
            classes_path = self.model_path.with_suffix('.txt')
            if classes_path.exists():
                with open(classes_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                self.logger.info(f"Nomes de classes carregados: {len(self.class_names)} classes")
            
            # Se não encontrou arquivo de classes, tentar extrair do modelo
            elif hasattr(self.model, 'names'):
                self.class_names = self.model.names
                self.logger.info(f"Nomes de classes extraídos do modelo: {len(self.class_names)} classes")
            
            # Se não encontrou classes, usar lista padrão para buffet
            else:
                self.class_names = ["prato_vazio", "prato_cheio", "alimento_a", "alimento_b", "alimento_c"]
                self.logger.warning(f"Usando nomes de classes padrão: {self.class_names}")
            
        except Exception as e:
            self.logger.warning(f"Erro ao carregar nomes das classes: {e}")
            self.class_names = [f"classe_{i}" for i in range(80)]  # Padrão COCO
    
    def preprocess_image(self, image):
        """
        Pré-processa uma imagem para inferência.
        
        Args:
            image: Imagem no formato numpy array (BGR)
            
        Returns:
            Tensor processado para inferência
        """
        try:
            # Verificar se a imagem é válida
            if image is None or image.size == 0:
                self.logger.error("Imagem inválida recebida para pré-processamento")
                return None
            
            # Se usando a API Ultralytics YOLO, não é necessário pré-processar
            if hasattr(self.model, 'predict'):
                return image
            
            # Pré-processamento para modelo PyTorch puro
            # Converter BGR para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para o tamanho esperado pelo modelo
            img_size = 640  # Tamanho padrão para YOLOv11x
            resized = cv2.resize(image_rgb, (img_size, img_size))
            
            # Normalizar e converter para tensor
            img_tensor = torch.from_numpy(resized).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
            
            # Mover para o dispositivo correto
            img_tensor = img_tensor.to(self.device)
            
            return img_tensor
            
        except Exception as e:
            self.logger.error(f"Erro durante o pré-processamento da imagem: {e}")
            return None
    
    def process_frame(self, frame):
        """
        Processa um único frame para detecção de objetos.
        
        Args:
            frame: Imagem no formato numpy array (BGR)
            
        Returns:
            Lista de detecções no formato:
            [
                {
                    'class_id': int,
                    'class_name': str,
                    'confidence': float,
                    'box': [x1, y1, x2, y2]
                },
                ...
            ]
        """
        try:
            if frame is None or frame.size == 0:
                self.logger.warning("Frame vazio recebido para processamento")
                return []
            
            # Pré-processar imagem
            input_tensor = self.preprocess_image(frame)
            if input_tensor is None:
                return []
            
            # Realizar inferência
            if hasattr(self.model, 'predict'):
                # API Ultralytics YOLO
                with torch.no_grad():
                    results = self.model.predict(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device=self.device
                    )
                
                # Processar resultados da API Ultralytics
                detections = []
                for result in results:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        box_xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"classe_{cls_id}"
                        
                        detections.append({
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': conf,
                            'box': box_xyxy.tolist()
                        })
                
                return detections
            
            else:
                # Modelo PyTorch puro
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                # Parse do output do modelo PyTorch
                # Nota: O formato exato pode variar dependendo da implementação do YOLOv11x
                # Este é um processamento genérico
                detections = []
                
                # Verificar formatos de saída comuns
                if isinstance(output, tuple):
                    output = output[0]  # Geralmente o primeiro elemento contém as detecções
                
                if isinstance(output, list):
                    output = output[0]  # Primeiro elemento para batch size=1
                
                # Processar saída
                if output.ndim == 3:  # [batch, num_detections, attrs]
                    for det in output[0]:
                        if len(det) >= 6:  # x1, y1, x2, y2, conf, cls_id, ...
                            x1, y1, x2, y2, conf, cls_id = det[:6]
                            if conf >= self.conf_threshold:
                                cls_id = int(cls_id.item())
                                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"classe_{cls_id}"
                                
                                detections.append({
                                    'class_id': cls_id,
                                    'class_name': cls_name,
                                    'confidence': float(conf.item()),
                                    'box': [float(x1.item()), float(y1.item()), 
                                           float(x2.item()), float(y2.item())]
                                })
                
                return detections
            
        except Exception as e:
            self.logger.exception(f"Erro durante processamento do frame: {e}")
            return []
    
    def process_frames(self, frames_dict):
        """
        Processa múltiplos frames de diferentes câmeras.
        
        Args:
            frames_dict: Dicionário com {camera_id: frame}
            
        Returns:
            Dicionário com {camera_id: detections}
        """
        results = {}
        
        for camera_id, frame in frames_dict.items():
            self.logger.debug(f"Processando frame da câmera {camera_id}")
            detections = self.process_frame(frame)
            results[camera_id] = detections
            self.logger.debug(f"Câmera {camera_id}: {len(detections)} detecções")
        
        return results
    
    def draw_results(self, frame, detections):
        """
        Desenha os resultados de detecção em uma imagem.
        
        Args:
            frame: Imagem original no formato numpy array (BGR)
            detections: Lista de detecções do método process_frame
            
        Returns:
            Imagem com as detecções desenhadas
        """
        try:
            if frame is None or len(detections) == 0:
                return frame
            
            # Criar uma cópia da imagem
            image_copy = frame.copy()
            
            # Definir cores para diferentes classes
            colors = {
                'prato_vazio': (0, 0, 255),      # Vermelho
                'prato_cheio': (0, 255, 0),      # Verde
                'alimento_a': (255, 0, 0),       # Azul
                'alimento_b': (0, 255, 255),     # Amarelo
                'alimento_c': (255, 0, 255)      # Magenta
            }
            
            # Desenhar cada detecção
            for detection in detections:
                # Extrair informações
                box = detection['box']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Converter para inteiros
                x1, y1, x2, y2 = map(int, box)
                
                # Definir cor (ou usar padrão se não estiver no dicionário)
                color = colors.get(class_name, (255, 255, 255))
                
                # Desenhar retângulo
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
                
                # Desenhar texto
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(image_copy, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return image_copy
            
        except Exception as e:
            self.logger.error(f"Erro ao desenhar resultados: {e}")
            return frame
    
    def estimate_food_amount(self, detections):
        """
        Estima a quantidade de comida com base nas detecções.
        
        Args:
            detections: Lista de detecções do método process_frame
            
        Returns:
            Dicionário com estimativas de quantidade por classe
        """
        try:
            estimates = {}
            
            # Contar ocorrências de cada classe
            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                if class_name not in estimates:
                    estimates[class_name] = {
                        'count': 0,
                        'total_confidence': 0,
                        'total_area': 0
                    }
                
                # Incrementar contagem
                estimates[class_name]['count'] += 1
                estimates[class_name]['total_confidence'] += confidence
                
                # Calcular área
                box = detection['box']
                area = (box[2] - box[0]) * (box[3] - box[1])
                estimates[class_name]['total_area'] += area
            
            # Calcular médias
            for class_name in estimates:
                count = estimates[class_name]['count']
                estimates[class_name]['avg_confidence'] = estimates[class_name]['total_confidence'] / count
                estimates[class_name]['avg_area'] = estimates[class_name]['total_area'] / count
            
            return estimates
            
        except Exception as e:
            self.logger.error(f"Erro ao estimar quantidade de comida: {e}")
            return {}


class FoodVolumeEstimator:
    """
    Classe para estimar o volume/quantidade de comida nos pratos.
    Complementa o YOLOProcessor fornecendo métricas mais avançadas.
    """
    
    def __init__(self, pixel_to_cm_ratio=0.1):
        """
        Inicializa o estimador de volume.
        
        Args:
            pixel_to_cm_ratio: Razão de conversão de pixels para centímetros
        """
        self.logger = logging.getLogger(__name__)
        self.pixel_to_cm_ratio = pixel_to_cm_ratio
    
    def estimate_food_volume(self, detections, frame_height, frame_width):
        """
        Estima o volume de comida com base nas detecções.
        
        Args:
            detections: Lista de detecções do YOLOProcessor
            frame_height: Altura do frame em pixels
            frame_width: Largura do frame em pixels
            
        Returns:
            Dicionário com estimativas de volume e ocupação
        """
        try:
            # Calcular área total do frame
            total_area = frame_height * frame_width
            
            # Inicializar estimativas
            estimates = {
                'pratos_vazios': 0,
                'pratos_cheios': 0,
                'alimentos': {},
                'ocupacao_buffet': 0.0
            }
            
            # Processar detecções
            for detection in detections:
                class_name = detection['class_name']
                box = detection['box']
                
                # Calcular área em pixels
                area_px = (box[2] - box[0]) * (box[3] - box[1])
                
                # Calcular área em cm²
                area_cm2 = area_px * (self.pixel_to_cm_ratio ** 2)
                
                # Processar por tipo
                if class_name == 'prato_vazio':
                    estimates['pratos_vazios'] += 1
                elif class_name == 'prato_cheio':
                    estimates['pratos_cheios'] += 1
                elif class_name.startswith('alimento_'):
                    if class_name not in estimates['alimentos']:
                        estimates['alimentos'][class_name] = {
                            'count': 0,
                            'total_area_cm2': 0,
                            'percent_of_frame': 0
                        }
                    
                    # Incrementar contadores
                    estimates['alimentos'][class_name]['count'] += 1
                    estimates['alimentos'][class_name]['total_area_cm2'] += area_cm2
                    estimates['alimentos'][class_name]['percent_of_frame'] += (area_px / total_area) * 100
            
            # Calcular ocupação total do buffet
            total_food_area = 0
            for food in estimates['alimentos'].values():
                total_food_area += food['total_area_cm2']
            
            # Estimar ocupação do buffet (valor entre 0 e 1)
            estimates['ocupacao_buffet'] = min(1.0, total_food_area / (frame_width * frame_height * (self.pixel_to_cm_ratio ** 2)))
            
            return estimates
            
        except Exception as e:
            self.logger.error(f"Erro ao estimar volume de comida: {e}")
            return {}
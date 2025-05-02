#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Processamento - Sistema de Monitoramento de Buffet
Responsável pelo processamento das detecções do modelo YOLO,
desenho de frames para visualização e cálculo de porcentagem de ocupação.
"""

import logging
import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from threading import Lock
from datetime import datetime


class DetectionProcessor:
    """
    Classe para processar as detecções do modelo YOLO e calcular métricas.
    """
    
    def __init__(self, data_file="buffet_data.json", camera_config_file="config/cameras_novo.json"):
        """
        Inicializa o processador de detecções.
        
        Args:
            data_file: Caminho para o arquivo JSON onde os dados serão salvos
            camera_config_file: Caminho para o arquivo de configuração das câmeras
        """
        self.logger = logging.getLogger(__name__)
        self.max_areas = {}  # Armazenar a área máxima para cada classe/ID de objeto
        self.max_areas_lock = Lock()  # Lock para acesso concorrente ao dicionário de áreas máximas
        self.data_file = data_file
        self.data_file_lock = Lock()  # Lock para acesso concorrente ao arquivo de dados
        self.camera_config_file = camera_config_file
        self.camera_info = self._load_camera_config()
        self.color_map = {
            # Cores para diferentes classes (RGB)
            # Pode ser personalizado conforme necessário
            'food_tray': (0, 255, 0),     # Verde para bandejas de comida
            'empty_tray': (0, 0, 255),    # Vermelho para bandejas vazias
            'low_food': (0, 165, 255),    # Laranja para pouca comida
            'medium_food': (0, 255, 255), # Amarelo para comida média
            'full_food': (0, 255, 0),     # Verde para comida cheia
            'default': (255, 255, 255)    # Branco para classes não especificadas
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.line_thickness = 2
        
        # Inicializar arquivo de dados se não existir
        self._init_data_file()
        
        self.logger.debug("DetectionProcessor inicializado")
    
    def _load_camera_config(self):
        """
        Carrega as informações de configuração das câmeras, incluindo o restaurant.
        
        Returns:
            dict: Dicionário com informações das câmeras indexado por camera_id
        """
        camera_info = {}
        try:
            if os.path.exists(self.camera_config_file):
                with open(self.camera_config_file, 'r') as f:
                    camera_config = json.load(f)
                    
                for camera in camera_config.get('cameras', []):
                    camera_id = camera.get('id')
                    if camera_id:
                        camera_info[camera_id] = {
                            'restaurant': camera.get('restaurant', 'default'),
                            'location': camera.get('location', ''),
                            'ip': camera.get('ip', ''),
                            'port': camera.get('port', 80)
                        }
            else:
                self.logger.warning(f"Arquivo de configuração de câmeras não encontrado: {self.camera_config_file}")
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração das câmeras: {e}")
            
        return camera_info
        
    def _init_data_file(self):
        """
        Inicializa o arquivo de dados JSON se ele não existir.
        """
        with self.data_file_lock:
            if not os.path.exists(self.data_file):
                try:
                    # Inicializar com a estrutura no formato de buffet_data_exemplo.json
                    initial_data = {
                        "address": []
                    }
                    with open(self.data_file, 'w') as f:
                        json.dump(initial_data, f, indent=4)
                    self.logger.info(f"Arquivo de dados criado: {self.data_file}")
                except Exception as e:
                    self.logger.error(f"Erro ao criar arquivo de dados: {e}")
    
    def process_detections(self, camera_id, frame, detections, track_max_area=True):
        """
        Processa as detecções para um frame específico e gera um frame com anotações.
        
        Args:
            camera_id: ID da câmera/vídeo
            frame: Frame original (numpy array)
            detections: Lista de dicionários com detecções de objetos
            track_max_area: Se True, rastreia a área máxima para cada objeto detectado
            
        Returns:
            tuple: (frame_anotado, estatísticas)
        """
        if frame is None:
            self.logger.warning(f"Frame nulo recebido de {camera_id}")
            return None, {}
            
        # Clonar o frame para não modificar o original
        annotated_frame = frame.copy()
        
        # Estatísticas para este frame
        stats = {
            'total_detections': len(detections),
            'classes': {},
            'area_percentages': {},
            'needs_refill': []
        }
        
        # Processar cada detecção
        for i, detection in enumerate(detections):
            # Extrair informações da detecção
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            
            # Calcular identificador único para este objeto
            object_id = f"{camera_id}_{class_name}_{i}"
            
            # Desenhar bounding box e informações
            annotated_frame = self.draw_detection(
                annotated_frame, class_name, confidence, bbox, object_id
            )
            
            # Calcular área do objeto
            area = self.calculate_area(bbox)
            
            # Atualizar área máxima se necessário
            area_percentage = 0.0
            if track_max_area and area > 0:
                area_percentage = self.update_max_area(object_id, class_name, area)
                
                # Adicionar à lista de necessidade de reposição se a porcentagem for baixa
                if area_percentage < 0.3:
                    stats['needs_refill'].append({
                        'id': object_id, 
                        'class': class_name, 
                        'percentage': area_percentage
                    })
            
            # Atualizar estatísticas
            if class_name in stats['classes']:
                stats['classes'][class_name] += 1
            else:
                stats['classes'][class_name] = 1
            
            stats['area_percentages'][object_id] = area_percentage
            
            # Salvar dados no arquivo JSON
            dish_id = f"{camera_id}_{class_name}"
            self.save_area_percentage(camera_id, dish_id, class_name, area_percentage)
        
        return annotated_frame, stats
    
    def draw_detection(self, frame, class_name, confidence, bbox, object_id):
        """
        Desenha uma detecção no frame com bounding box, classe e porcentagem.
        
        Args:
            frame: Frame para desenhar (numpy array)
            class_name: Nome da classe
            confidence: Confiança da detecção
            bbox: Bounding box [x1, y1, x2, y2]
            object_id: ID único do objeto
            
        Returns:
            numpy.ndarray: Frame com anotações
        """
        # Converter coordenadas para inteiros
        x1, y1, x2, y2 = map(int, bbox)
        
        # Obter cor para esta classe
        color = self.color_map.get(class_name, self.color_map['default'])
        
        # Inverter BGR para RGB (OpenCV usa BGR)
        color_bgr = (color[2], color[1], color[0])
        
        # Desenhar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, self.line_thickness)
        
        # Obter porcentagem de área se disponível
        percentage_text = ""
        with self.max_areas_lock:
            if object_id in self.max_areas:
                area = self.calculate_area(bbox)
                max_area = self.max_areas[object_id]['max_area']
                if max_area > 0:
                    percentage = min(area / max_area, 1.0) * 100
                    percentage_text = f" ({percentage:.1f}%)"
        
        # Preparar texto com o nome da classe (prato) e porcentagem
        text = f"{class_name}{percentage_text}"
        
        # Obter dimensões do texto para posicionamento
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        text_width, text_height = text_size
        
        # Desenhar fundo para o texto
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width, y1), 
            color_bgr, 
            -1  # Preenchido
        )
        
        # Desenhar texto
        cv2.putText(
            frame, 
            text, 
            (x1, y1 - 5), 
            self.font, 
            self.font_scale, 
            (0, 0, 0),  # Preto
            self.font_thickness
        )
        
        return frame
    
    def calculate_area(self, bbox):
        """
        Calcula a área de um bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            float: Área do bounding box
        """
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height
    
    def update_max_area(self, object_id, class_name, current_area):
        """
        Atualiza a área máxima registrada para um objeto e retorna a porcentagem atual.
        
        Args:
            object_id: ID único do objeto
            class_name: Nome da classe
            current_area: Área atual do objeto
            
        Returns:
            float: Porcentagem da área atual em relação à área máxima (0.0 a 1.0)
        """
        percentage = 1.0  # Valor padrão
        
        with self.max_areas_lock:
            # Se este objeto ainda não foi registrado
            if object_id not in self.max_areas:
                self.max_areas[object_id] = {
                    'class': class_name,
                    'max_area': current_area,
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
                percentage = 1.0
            else:
                # Atualizar timestamp de última visualização
                self.max_areas[object_id]['last_seen'] = time.time()
                
                # Verificar se a área atual é maior que a máxima registrada
                current_max = self.max_areas[object_id]['max_area']
                
                if current_area > current_max:
                    # Atualizar área máxima
                    self.max_areas[object_id]['max_area'] = current_area
                    percentage = 1.0
                else:
                    # Calcular porcentagem
                    percentage = current_area / current_max if current_max > 0 else 1.0
        
        return percentage
    
    def save_area_percentage(self, camera_id, dish_id, dish_name, percentage):
        """
        Salva a porcentagem de área para um prato no arquivo JSON no formato buffet_data_exemplo.json.
        
        Args:
            camera_id: ID da câmera/vídeo
            dish_id: ID único do prato
            dish_name: Nome do prato (classe)
            percentage: Porcentagem atual (0.0 a 1.0)
        """
        timestamp = time.time()
        iso_timestamp = datetime.now().isoformat()
        needs_reposition = percentage < 0.3
        
        try:
            with self.data_file_lock:
                # Carregar dados existentes
                data = {}
                if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                    try:
                        with open(self.data_file, 'r') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.error(f"Erro ao decodificar {self.data_file}, criando novo arquivo")
                        data = {"address": []}
                
                # Verificar se a estrutura básica existe
                if "address" not in data:
                    data["address"] = []
                
                # Obter informações do restaurante para esta câmera
                restaurant_id = "default"
                if camera_id in self.camera_info:
                    restaurant_id = self.camera_info[camera_id].get("restaurant", "default")
                
                # Verificar se o restaurante já existe
                restaurant_index = None
                for i, restaurant in enumerate(data["address"]):
                    if restaurant.get("restaurant") == restaurant_id:
                        restaurant_index = i
                        break
                
                # Se o restaurante não existir, criar um novo
                if restaurant_index is None:
                    new_restaurant = {
                        "restaurant": restaurant_id,
                        "locations": []
                    }
                    data["address"].append(new_restaurant)
                    restaurant_index = len(data["address"]) - 1
                
                # Verificar se a localização já existe dentro do restaurante
                location_index = None
                restaurant_locations = data["address"][restaurant_index].get("locations", [])
                for i, location in enumerate(restaurant_locations):
                    if location.get("location_id") == camera_id:
                        location_index = i
                        break
                
                # Se a localização não existir, criar uma nova
                if location_index is None:
                    # Buscar o location_name ou usar um padrão
                    location_name = self.get_location_name(camera_id)
                    
                    new_location = {
                        "location_id": camera_id,
                        "location_name": location_name,
                        "dishes": []
                    }
                    
                    # Garantir que a chave "locations" existe
                    if "locations" not in data["address"][restaurant_index]:
                        data["address"][restaurant_index]["locations"] = []
                        
                    data["address"][restaurant_index]["locations"].append(new_location)
                    location_index = len(data["address"][restaurant_index]["locations"]) - 1
                
                # Verificar se o prato já existe na localização
                dish_index = None
                for i, dish in enumerate(data["address"][restaurant_index]["locations"][location_index]["dishes"]):
                    if dish.get("dish_id") == dish_id:
                        dish_index = i
                        break
                
                # Criar ou atualizar as informações do prato
                dish_data = {
                    "dish_id": dish_id,
                    "dish_name": dish_name,
                    "percentage_remaining": int(percentage * 100),
                    "needs_reposition": needs_reposition,
                    "timestamp": iso_timestamp
                }
                
                # Se o prato já existir, atualizá-lo
                if dish_index is not None:
                    data["address"][restaurant_index]["locations"][location_index]["dishes"][dish_index] = dish_data
                else:
                    # Se não existir, adicioná-lo
                    data["address"][restaurant_index]["locations"][location_index]["dishes"].append(dish_data)
                
                # Salvar os dados atualizados
                with open(self.data_file, 'w') as f:
                    json.dump(data, f, indent=4)
                
                # Log a cada X salvamentos para não sobrecarregar o log
                if timestamp % 60 < 1:  # Aproximadamente uma vez por minuto
                    self.logger.debug(f"Dados atualizados para {camera_id}/{dish_name}")
                    
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados no arquivo JSON: {e}")
    
    def get_location_name(self, camera_id):
        """
        Obtém o nome da localização com base no ID da câmera.
        Esta função pode ser expandida para buscar o nome real da localização 
        a partir de um arquivo de configuração.
        
        Args:
            camera_id: ID da câmera
            
        Returns:
            str: Nome da localização
        """
        # Para simplificar, usamos um mapeamento básico
        # Em uma implementação real, isso poderia vir de uma configuração
        location_names = {
            "cam1": "Buffet Principal",
            "cam2": "Buffet de Entradas",
            "cam3": "Buffet de Saladas",
            "cam4": "Buffet de Pratos Quentes",
            "cam5": "Buffet de Sobremesas",
            "cam6": "Buffet Vegetariano",
            "cam7": "Buffet de Frutas",
            "cam8": "Buffet de Bebidas",
            "cam9": "Buffet Infantil",
            "cam10": "Buffet de Sopas"
        }
        
        return location_names.get(camera_id, f"Local {camera_id}")
    
    def load_area_data(self, restaurant_id=None, camera_id=None, dish_id=None):
        """
        Carrega os dados de área do arquivo JSON.
        
        Args:
            restaurant_id: ID do restaurante para filtrar (opcional)
            camera_id: ID da câmera para filtrar (opcional)
            dish_id: ID do prato para filtrar (opcional)
            
        Returns:
            dict: Dados carregados do arquivo
        """
        try:
            with self.data_file_lock:
                if not os.path.exists(self.data_file):
                    return {"address": []}
                    
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Se não houver filtros, retornar todos os dados
                if restaurant_id is None and camera_id is None and dish_id is None:
                    return data
                
                # Filtrar por restaurant_id, camera_id e dish_id
                filtered_data = {"address": []}
                
                for restaurant in data.get("address", []):
                    # Se tiver filtro de restaurant_id e não for o restaurante correto, pular
                    if restaurant_id is not None and restaurant.get("restaurant") != restaurant_id:
                        continue
                    
                    # Criar cópia filtrada do restaurante
                    filtered_restaurant = {"restaurant": restaurant.get("restaurant"), "locations": []}
                    
                    # Se não tiver filtro de camera_id, incluir todo o restaurante
                    if camera_id is None and dish_id is None:
                        filtered_restaurant["locations"] = restaurant.get("locations", [])
                        filtered_data["address"].append(filtered_restaurant)
                        continue
                    
                    # Filtrar por camera_id
                    for location in restaurant.get("locations", []):
                        # Se tiver filtro de camera_id e não for a localização correta, pular
                        if camera_id is not None and location.get("location_id") != camera_id:
                            continue
                        
                        # Se não tiver filtro de dish_id, incluir toda a localização
                        if dish_id is None:
                            filtered_restaurant["locations"].append(location)
                            continue
                        
                        # Filtrar por dish_id
                        filtered_location = location.copy()
                        filtered_location["dishes"] = [
                            dish for dish in location.get("dishes", [])
                            if dish.get("dish_id") == dish_id
                        ]
                        
                        if filtered_location["dishes"]:
                            filtered_restaurant["locations"].append(filtered_location)
                    
                    # Adicionar o restaurante filtrado se tiver localizações
                    if filtered_restaurant["locations"]:
                        filtered_data["address"].append(filtered_restaurant)
                
                return filtered_data
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados do arquivo JSON: {e}")
            return {"address": []}
    
    def clean_old_records(self, max_age_seconds=3600):
        """
        Remove registros de objetos que não são vistos há um determinado tempo.
        
        Args:
            max_age_seconds: Tempo máximo em segundos para manter registros inativos
        """
        current_time = time.time()
        objects_to_remove = []
        
        with self.max_areas_lock:
            for object_id, data in self.max_areas.items():
                last_seen = data['last_seen']
                if current_time - last_seen > max_age_seconds:
                    objects_to_remove.append(object_id)
            
            # Remover objetos antigos
            for object_id in objects_to_remove:
                del self.max_areas[object_id]
                
        if objects_to_remove:
            self.logger.debug(f"Removidos {len(objects_to_remove)} registros antigos")
    
    def get_status_summary(self, restaurant_id=None, camera_id=None):
        """
        Obtém um resumo de status dos objetos rastreados.
        
        Args:
            restaurant_id: Se fornecido, filtra apenas objetos deste restaurante
            camera_id: Se fornecido, filtra apenas objetos desta câmera
            
        Returns:
            dict: Resumo do status
        """
        # Carregar dados do arquivo JSON diretamente para usar a nova estrutura
        data = self.load_area_data(restaurant_id, camera_id)
        
        summary = {
            'total_restaurants': len(data.get("address", [])),
            'total_locations': 0,
            'total_dishes': 0,
            'dishes_by_restaurant': {},
            'dishes_by_location': {},
            'needs_reposition': []
        }
        
        # Processar dados de acordo com a nova estrutura
        for restaurant in data.get("address", []):
            restaurant_id = restaurant.get("restaurant")
            locations = restaurant.get("locations", [])
            summary['total_locations'] += len(locations)
            
            # Inicializar contagem de pratos para este restaurante
            if restaurant_id not in summary['dishes_by_restaurant']:
                summary['dishes_by_restaurant'][restaurant_id] = 0
            
            # Processar cada localização do restaurante
            for location in locations:
                location_id = location.get("location_id")
                dishes = location.get("dishes", [])
                summary['total_dishes'] += len(dishes)
                
                # Incrementar contagem de pratos para este restaurante
                summary['dishes_by_restaurant'][restaurant_id] += len(dishes)
                
                # Contagem de pratos por localização
                summary['dishes_by_location'][f"{restaurant_id}_{location_id}"] = len(dishes)
                
                # Verificar pratos que precisam de reposição
                for dish in dishes:
                    if dish.get("needs_reposition", False):
                        summary['needs_reposition'].append({
                            'restaurant': restaurant_id,
                            'location_id': location_id,
                            'dish_id': dish.get("dish_id"),
                            'dish_name': dish.get("dish_name"),
                            'percentage': dish.get("percentage_remaining", 0) / 100.0
                        })
        
        return summary


class FrameProcessor:
    """
    Classe para processamento de frames, incluindo redimensionamento,
    concatenação de múltiplos frames e outras operações visuais.
    """
    
    def __init__(self):
        """
        Inicializa o processador de frames.
        """
        self.logger = logging.getLogger(__name__)
        
    def resize_frame(self, frame, target_width=None, target_height=None):
        """
        Redimensiona um frame para as dimensões desejadas.
        
        Args:
            frame: Frame para redimensionar
            target_width: Largura desejada (se None, mantém a proporção)
            target_height: Altura desejada (se None, mantém a proporção)
            
        Returns:
            numpy.ndarray: Frame redimensionado
        """
        if frame is None:
            return None
            
        original_height, original_width = frame.shape[:2]
        
        # Se ambas as dimensões forem None, retorna o frame original
        if target_width is None and target_height is None:
            return frame
            
        # Calcular as dimensões mantendo a proporção
        if target_width is None:
            aspect_ratio = original_width / original_height
            target_width = int(target_height * aspect_ratio)
        elif target_height is None:
            aspect_ratio = original_height / original_width
            target_height = int(target_width * aspect_ratio)
            
        # Redimensionar o frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        return resized_frame
    
    def concat_frames(self, frames, layout=None):
        """
        Concatena múltiplos frames em um único frame para visualização.
        
        Args:
            frames: Lista de frames para concatenar
            layout: Tupla (rows, cols) para o layout da grade, 
                   se None, tenta criar um layout quadrado
            
        Returns:
            numpy.ndarray: Frame combinado
        """
        if not frames:
            return None
            
        # Filtrar frames None
        valid_frames = [f for f in frames if f is not None]
        
        if not valid_frames:
            return None
            
        num_frames = len(valid_frames)
        
        # Determinar layout se não fornecido
        if layout is None:
            cols = int(np.ceil(np.sqrt(num_frames)))
            rows = int(np.ceil(num_frames / cols))
        else:
            rows, cols = layout
            
        # Verificar se temos frames suficientes para o layout
        if rows * cols < num_frames:
            self.logger.warning(f"Layout {rows}x{cols} não comporta {num_frames} frames")
            cols = int(np.ceil(np.sqrt(num_frames)))
            rows = int(np.ceil(num_frames / cols))
            
        # Encontrar as dimensões máximas
        max_height = max(frame.shape[0] for frame in valid_frames)
        max_width = max(frame.shape[1] for frame in valid_frames)
        
        # Criar frame combinado
        combined_frame = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)
        
        # Preencher o frame combinado
        for i, frame in enumerate(valid_frames):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            # Calcular posição para este frame
            y_offset = row * max_height
            x_offset = col * max_width
            
            # Redimensionar frame para o tamanho máximo
            resized = self.resize_frame(frame, max_width, max_height)
            
            # Copiar para o frame combinado
            combined_frame[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
            
        return combined_frame
    
    def add_timestamp(self, frame, camera_id=None):
        """
        Adiciona timestamp (e opcionalmente ID da câmera) ao frame.
        
        Args:
            frame: Frame para adicionar o timestamp
            camera_id: ID da câmera (opcional)
            
        Returns:
            numpy.ndarray: Frame com timestamp
        """
        if frame is None:
            return None
            
        # Clonar frame para não modificar o original
        annotated_frame = frame.copy()
        
        # Preparar o texto
        timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
        if camera_id:
            timestamp_text = f"{camera_id} | {timestamp_text}"
            
        # Desenhar texto
        cv2.putText(
            annotated_frame,
            timestamp_text,
            (10, 30),  # Posição
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Escala
            (255, 255, 255),  # Cor (branco)
            2  # Espessura
        )
        
        return annotated_frame
    
    def draw_food_percentage(self, frame, percentage, position=None):
        """
        Desenha um indicador visual da porcentagem de comida.
        
        Args:
            frame: Frame para adicionar o indicador
            percentage: Porcentagem de comida (0.0 a 1.0)
            position: Tupla (x, y) com a posição do indicador, se None, usa o canto superior direito
            
        Returns:
            numpy.ndarray: Frame com indicador
        """
        if frame is None:
            return None
            
        # Clonar frame para não modificar o original
        annotated_frame = frame.copy()
        
        # Definir posição se não fornecida
        if position is None:
            frame_height, frame_width = frame.shape[:2]
            position = (frame_width - 210, 30)
            
        x, y = position
        
        # Desenhar barra de fundo
        bar_width = 200
        bar_height = 20
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (100, 100, 100),  # Cinza escuro
            -1  # Preenchido
        )
        
        # Determinar cor baseada na porcentagem
        if percentage < 0.3:
            color = (0, 0, 255)  # Vermelho
        elif percentage < 0.7:
            color = (0, 165, 255)  # Laranja
        else:
            color = (0, 255, 0)  # Verde
            
        # Desenhar barra de progresso
        filled_width = int(bar_width * percentage)
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + filled_width, y + bar_height),
            color,
            -1  # Preenchido
        )
        
        # Desenhar borda
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (255, 255, 255),  # Branco
            1  # Espessura
        )
        
        # Desenhar texto
        text = f"{percentage * 100:.1f}%"
        cv2.putText(
            annotated_frame,
            text,
            (x + 5, y + 15),  # Posição
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Escala
            (255, 255, 255),  # Cor (branco)
            1  # Espessura
        )
        
        return annotated_frame


# Função para teste simples do módulo quando executado diretamente
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print("=== Teste do Módulo de Processamento ===")
    
    # Criar processadores
    detection_processor = DetectionProcessor()
    frame_processor = FrameProcessor()
    
    # Criar frame de teste
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Frame de Teste", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Criar detecções de teste
    test_detections = [
        {
            'class': 'food_tray',
            'confidence': 0.95,
            'bbox': [100, 100, 300, 200]
        },
        {
            'class': 'low_food',
            'confidence': 0.85,
            'bbox': [350, 150, 500, 250]
        }
    ]
    
    # Processar detecções
    annotated_frame, stats = detection_processor.process_detections("test", test_frame, test_detections)
    
    # Adicionar timestamp
    annotated_frame = frame_processor.add_timestamp(annotated_frame, "test")
    
    # Desenhar porcentagem
    annotated_frame = frame_processor.draw_food_percentage(annotated_frame, 0.75)
    
    # Carregar e mostrar dados salvos
    saved_data = detection_processor.load_area_data()
    print(f"Dados salvos: {saved_data}")
    
    print("Processamento de teste concluído!")
    print(f"Estatísticas: {stats}")
    print("\n=== Teste finalizado ===")
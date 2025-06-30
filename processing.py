#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Processamento - Sistema de Monitoramento de Buffet
Modificação simples: mesmos pratos de câmeras diferentes compartilham área máxima
e apenas a câmera com maior porcentagem grava no JSON.
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
    Modificada para consolidar área máxima entre câmeras do mesmo prato.
    """
    
    def __init__(self, camera_info, data_file="buffet_data.json"):
        """
        Inicializa o processador de detecções.
        
        Args:
            camera_info: Dicionário com informações das câmeras indexado por camera_id
            data_file: Caminho para o arquivo JSON onde os dados serão salvos
        """
        self.logger = logging.getLogger(__name__)
        
        self.max_areas = {}  # Chave: dish_name, Valor: dados da área máxima
        self.max_areas_lock = Lock()
        
        self.best_cameras = {}  # Chave: dish_name, Valor: dados da melhor câmera
        self.best_cameras_lock = Lock()
        
        self.data_file = data_file
        self.data_file_lock = Lock()
        self.camera_info = camera_info
        
        self.color_map = {
            'food_tray': (0, 255, 0),
            'empty_tray': (0, 0, 255),
            'low_food': (0, 165, 255),
            'medium_food': (0, 255, 255),
            'full_food': (0, 255, 0),
            'default': (255, 255, 255)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.line_thickness = 2
        
        self._init_data_file()
        
        self.logger.debug("DetectionProcessor inicializado com consolidação de área máxima")
        
    def _init_data_file(self):
        """
        Inicializa o arquivo de dados JSON se ele não existir.
        """
        with self.data_file_lock:
            if not os.path.exists(self.data_file):
                try:
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
        """
        if frame is None:
            self.logger.warning(f"Frame nulo recebido de {camera_id}")
            return None, {}
            
        annotated_frame = frame.copy()
        
        stats = {
            'total_detections': len(detections),
            'classes': {},
            'area_percentages': {},
            'needs_refill': []
        }
        
        for i, detection in enumerate(detections):
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            
            object_id = f"{camera_id}_{class_name}_{i}"
            
            annotated_frame = self.draw_detection(
                annotated_frame, class_name, confidence, bbox, object_id
            )
            
            area = self.calculate_area(bbox)
            
            area_percentage = 0.0
            if track_max_area and area > 0:
                area_percentage = self.update_consolidated_max_area(class_name, camera_id, area)
                
                if area_percentage < 0.3:
                    stats['needs_refill'].append({
                        'id': object_id, 
                        'class': class_name, 
                        'percentage': area_percentage
                    })
            
            if class_name in stats['classes']:
                stats['classes'][class_name] += 1
            else:
                stats['classes'][class_name] = 1
            
            stats['area_percentages'][object_id] = area_percentage
            
            self.save_if_best_camera(camera_id, class_name, area_percentage)
        
        return annotated_frame, stats
    
    def draw_detection(self, frame, class_name, confidence, bbox, object_id):
        """
        Desenha uma detecção no frame com bounding box, classe e porcentagem.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        color = self.color_map.get(class_name, self.color_map['default'])
        color_bgr = (color[2], color[1], color[0])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, self.line_thickness)
        
        percentage_text = ""
        with self.max_areas_lock:
            if class_name in self.max_areas:
                area = self.calculate_area(bbox)
                max_area = self.max_areas[class_name]['max_area']
                if max_area > 0:
                    percentage = min(area / max_area, 1.0) * 100
                    percentage_text = f" ({percentage:.1f}%)"
        
        text = f"{class_name}{percentage_text}"
        
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        text_width, text_height = text_size
        
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width, y1), 
            color_bgr, 
            -1
        )
        
        cv2.putText(
            frame, 
            text, 
            (x1, y1 - 5), 
            self.font, 
            self.font_scale, 
            (0, 0, 0),
            self.font_thickness
        )
        
        return frame
    
    def calculate_area(self, bbox):
        """
        Calcula a área de um bounding box.
        """
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height
    
    def update_consolidated_max_area(self, dish_name, camera_id, current_area):
        """
        Atualiza a área máxima consolidada para um prato e retorna a porcentagem atual.
        """
        percentage = 1.0
        
        with self.max_areas_lock:
            if dish_name not in self.max_areas:
                self.max_areas[dish_name] = {
                    'max_area': current_area,
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
                percentage = 1.0
                self.logger.debug(f"Novo prato registrado: {dish_name} (área: {current_area})")
            else:
                self.max_areas[dish_name]['last_seen'] = time.time()
                current_max = self.max_areas[dish_name]['max_area']
                
                if current_area > current_max:
                    self.max_areas[dish_name]['max_area'] = current_area
                    percentage = 1.0
                    self.logger.info(f"Nova área máxima global para {dish_name}: {current_area} (câmera {camera_id})")
                else:
                    percentage = current_area / current_max if current_max > 0 else 1.0
        
        with self.best_cameras_lock:
            if dish_name not in self.best_cameras or percentage > self.best_cameras[dish_name]['percentage']:
                self.best_cameras[dish_name] = {
                    'camera_id': camera_id,
                    'percentage': percentage,
                    'timestamp': time.time()
                }
        
        return percentage
    
    def save_if_best_camera(self, camera_id, dish_name, percentage):
        """
        Salva os dados apenas se esta câmera tem a maior porcentagem para este prato.
        """
        is_best_camera = False
        with self.best_cameras_lock:
            if dish_name in self.best_cameras:
                is_best_camera = self.best_cameras[dish_name]['camera_id'] == camera_id
        
        if is_best_camera:
            self.save_area_percentage(camera_id, dish_name, percentage)
    
    def save_area_percentage(self, camera_id, dish_name, percentage):
        """
        Salva a porcentagem de área para um prato no arquivo JSON de forma mais robusta e limpa.
        Utiliza escrita atômica para evitar corrupção de dados.
        """
        iso_timestamp = datetime.now().isoformat()
        
        try:
            with self.data_file_lock:
                data = {"address": []}
                if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                    try:
                        with open(self.data_file, 'r') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Arquivo JSON corrompido ({self.data_file}). Será sobrescrito.")
                
                camera_details = self.camera_info.get(camera_id, {})
                restaurant_id = camera_details.get("restaurant", "default")
                location_name = camera_details.get("location_name", f"Local {camera_id}")

                restaurant_obj = next((r for r in data.get("address", []) if r.get("restaurant") == restaurant_id), None)
                if not restaurant_obj:
                    restaurant_obj = {"restaurant": restaurant_id, "locations": []}
                    data["address"].append(restaurant_obj)

                location_obj = next((loc for loc in restaurant_obj.get("locations", []) if loc.get("location_id") == camera_id), None)
                if not location_obj:
                    location_obj = {
                        "location_id": camera_id,
                        "location_name": location_name,
                        "dishes": []
                    }
                    restaurant_obj["locations"].append(location_obj)

                dish_data = {
                    "dish_id": f"{camera_id}_{dish_name}",
                    "dish_name": dish_name,
                    "percentage_remaining": int(percentage * 100),
                    "needs_reposition": percentage < 0.5,
                    "timestamp": iso_timestamp
                }
                
                location_obj["dishes"] = [d for d in location_obj["dishes"] if d.get("dish_name") != dish_name]
                location_obj["dishes"].append(dish_data)

                temp_file_path = self.data_file + ".tmp"
                with open(temp_file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                os.replace(temp_file_path, self.data_file)
                    
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados no arquivo JSON: {e}", exc_info=True)
    
    def load_area_data(self, restaurant_id=None, camera_id=None, dish_id=None):
        """
        Carrega e filtra dados de área do arquivo JSON de forma mais eficiente.
        """
        try:
            with self.data_file_lock:
                if not os.path.exists(self.data_file):
                    return {"address": []}
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Erro ao carregar dados do arquivo JSON: {e}")
            return {"address": []}

        if not any([restaurant_id, camera_id, dish_id]):
            return data
            
        filtered_address = data.get("address", [])

        if restaurant_id:
            filtered_address = [r for r in filtered_address if r.get("restaurant") == restaurant_id]

        if camera_id:
            for restaurant in filtered_address:
                restaurant["locations"] = [loc for loc in restaurant.get("locations", []) if loc.get("location_id") == camera_id]
        
        if dish_id:
            for restaurant in filtered_address:
                for location in restaurant.get("locations", []):
                    location["dishes"] = [d for d in location.get("dishes", []) if d.get("dish_id") == dish_id]

        for r in filtered_address:
            r["locations"] = [loc for loc in r.get("locations", []) if loc.get("dishes") or not dish_id]
        
        filtered_address = [r for r in filtered_address if r.get("locations")]
        
        return {"address": filtered_address}
    
    def clean_old_records(self, max_age_seconds=3600):
        """
        Remove registros de objetos que não são vistos há um determinado tempo.
        """
        current_time = time.time()
        dishes_to_remove = []
        
        with self.max_areas_lock:
            for dish_name, data in self.max_areas.items():
                last_seen = data['last_seen']
                if current_time - last_seen > max_age_seconds:
                    dishes_to_remove.append(dish_name)
            
            for dish_name in dishes_to_remove:
                del self.max_areas[dish_name]
        
        with self.best_cameras_lock:
            for dish_name in dishes_to_remove:
                if dish_name in self.best_cameras:
                    del self.best_cameras[dish_name]
                
        if dishes_to_remove:
            self.logger.debug(f"Removidos {len(dishes_to_remove)} registros antigos")
    
    def get_status_summary(self, restaurant_id=None, camera_id=None):
        """
        Obtém um resumo de status dos objetos rastreados.
        """
        data = self.load_area_data(restaurant_id, camera_id)
        
        summary = {
            'total_restaurants': len(data.get("address", [])),
            'total_locations': 0,
            'total_dishes': 0,
            'dishes_by_restaurant': {},
            'dishes_by_location': {},
            'needs_reposition': []
        }
        
        for restaurant in data.get("address", []):
            restaurant_id = restaurant.get("restaurant")
            locations = restaurant.get("locations", [])
            summary['total_locations'] += len(locations)
            
            if restaurant_id not in summary['dishes_by_restaurant']:
                summary['dishes_by_restaurant'][restaurant_id] = 0
            
            for location in locations:
                location_id = location.get("location_id")
                dishes = location.get("dishes", [])
                summary['total_dishes'] += len(dishes)
                
                summary['dishes_by_restaurant'][restaurant_id] += len(dishes)
                summary['dishes_by_location'][f"{restaurant_id}_{location_id}"] = len(dishes)
                
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
        """
        if frame is None:
            return None
            
        original_height, original_width = frame.shape[:2]
        
        if target_width is None and target_height is None:
            return frame
            
        if target_width is None:
            aspect_ratio = original_width / original_height
            target_width = int(target_height * aspect_ratio)
        elif target_height is None:
            aspect_ratio = original_height / original_width
            target_height = int(target_width * aspect_ratio)
            
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        return resized_frame
    
    def concat_frames(self, frames, layout=None):
        """
        Concatena múltiplos frames em um único frame para visualização.
        """
        if not frames:
            return None
            
        valid_frames = [f for f in frames if f is not None]
        
        if not valid_frames:
            return None
            
        num_frames = len(valid_frames)
        
        if layout is None:
            cols = int(np.ceil(np.sqrt(num_frames)))
            rows = int(np.ceil(num_frames / cols))
        else:
            rows, cols = layout
            
        if rows * cols < num_frames:
            self.logger.warning(f"Layout {rows}x{cols} não comporta {num_frames} frames")
            cols = int(np.ceil(np.sqrt(num_frames)))
            rows = int(np.ceil(num_frames / cols))
            
        max_height = max(frame.shape[0] for frame in valid_frames)
        max_width = max(frame.shape[1] for frame in valid_frames)
        
        combined_frame = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)
        
        for i, frame in enumerate(valid_frames):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            y_offset = row * max_height
            x_offset = col * max_width
            
            resized = self.resize_frame(frame, max_width, max_height)
            
            combined_frame[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
            
        return combined_frame
    
    def add_timestamp(self, frame, camera_id=None):
        """
        Adiciona timestamp (e opcionalmente ID da câmera) ao frame.
        """
        if frame is None:
            return None
            
        annotated_frame = frame.copy()
        
        timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
        if camera_id:
            timestamp_text = f"{camera_id} | {timestamp_text}"
            
        cv2.putText(
            annotated_frame,
            timestamp_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
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
    # CORREÇÃO: Para testar, precisamos de um 'camera_info' mockado.
    mock_camera_info = {
        "test": {
            "restaurant": "mock_restaurant",
            "location_name": "Local de Teste"
        }
    }
    detection_processor = DetectionProcessor(camera_info=mock_camera_info)
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
    
    # O método draw_food_percentage foi removido por ser redundante.
    # A porcentagem já é exibida na bounding box da detecção.
    
    # Carregar e mostrar dados salvos
    saved_data = detection_processor.load_area_data()
    print(f"Dados salvos: {json.dumps(saved_data, indent=2)}")
    
    print("Processamento de teste concluído!")
    print(f"Estatísticas: {stats}")
    print("\n=== Teste finalizado ===")
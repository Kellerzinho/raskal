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

# Importação ajustada para a nova estrutura
from utils.dish_name_mapper import DishNameReplacer
from api_client import ExternalAPIClient


class DetectionProcessor:
    """
    Classe para processar as detecções do modelo YOLO e calcular métricas.
    Modificada para consolidar área máxima entre câmeras do mesmo prato.
    """
    
    def __init__(self, camera_info, data_file="buffet_data.json", dish_name_replacer=None, dashboard_url=None, auth_token=None):
        """
        Inicializa o processador de detecções.
        
        Args:
            camera_info: Dicionário com informações das câmeras indexado por camera_id
            data_file: Caminho para o arquivo JSON onde os dados serão salvos
            dish_name_replacer: Instância de DishNameReplacer para traduzir nomes de pratos
            dashboard_url: URL da API do dashboard externo
            auth_token: Token de autenticação para a API do dashboard
        """
        self.logger = logging.getLogger(__name__)
        
        self.max_areas = {}  # Chave: dish_name, Valor: dados da área máxima
        self.max_areas_lock = Lock()
        
        self.best_cameras = {}  # Chave: dish_name, Valor: dados da melhor câmera
        self.best_cameras_lock = Lock()
        
        self.data_file = "config/buffet_data.json"
        self.data_file_lock = Lock()
        self.camera_info = camera_info
        self.dish_name_replacer = dish_name_replacer or DishNameReplacer() # Fallback para o caso de não ser fornecido
        
        # Inicializa o cliente da API do dashboard se a URL for fornecida
        self.external_client = None
        if dashboard_url:
            self.external_client = ExternalAPIClient(dashboard_url, auth_token)
            self.logger.info(f"Cliente da API do dashboard inicializado com URL: {dashboard_url}")
        
        # Configurações de fonte e linha para as detecções
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.font_thickness = 2
        self.line_thickness = 3
        
        # Data de referência para reset diário
        self.current_date = datetime.now().date()
        
        self._init_data_file()
        
        self.logger.debug("DetectionProcessor inicializado com consolidação de área máxima")
        
    def _init_data_file(self):
        """
        Garante que o arquivo de dados JSON seja recriado a cada inicialização,
        removendo qualquer registro de execuções anteriores.
        """
        with self.data_file_lock:
            try:
                if os.path.exists(self.data_file):
                    os.remove(self.data_file)
                    self.logger.info(f"Arquivo de dados existente removido: {self.data_file}")
                
                initial_data = {
                    "address": []
                }
                with open(self.data_file, 'w') as f:
                    json.dump(initial_data, f, indent=4)
                self.logger.info(f"Arquivo de dados recriado com sucesso: {self.data_file}")
            except Exception as e:
                self.logger.error(f"Erro ao recriar o arquivo de dados: {e}")
    
    def process_detections(self, frame, camera_id, boxes, class_names):
        """
        Processa as detecções do YOLO, desenha no frame e calcula estatísticas.
        Agora agrupa por nome original de prato para salvar IDs corretos.
        """
        if frame is None:
            self.logger.warning(f"Frame nulo recebido de {camera_id}")
            return None, {}

        annotated_frame = frame.copy()
        
        stats = {
            'total_detections': 0,
            'classes': {},
            'area_percentages': {},
            'needs_refill': []
        }

        if boxes is None:
            return annotated_frame, stats

        stats['total_detections'] = len(boxes)

        # Dicionário para acumular áreas por nome de prato ORIGINAL
        dish_areas_by_original_name = {}  # key: original_class_name, value: total_area
        all_detections_by_original_name = {} # key: original_class_name, value: list of (bbox, confidence)


        # Primeira passagem: acumular áreas e informações de detecção por nome original
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            confidence = boxes.conf[i].item()
            class_id = int(boxes.cls[i].item())
            original_class_name = class_names[class_id]
            
            area = self.calculate_area(bbox)
            
            # Acumula a área total para este prato (pelo nome original)
            dish_areas_by_original_name.setdefault(original_class_name, 0)
            dish_areas_by_original_name[original_class_name] += area

            # Agrupa detecções pelo nome original
            all_detections_by_original_name.setdefault(original_class_name, [])
            all_detections_by_original_name[original_class_name].append((bbox, confidence))

            # Para estatísticas, usa o nome traduzido
            dish_name = self.dish_name_replacer.get_replacement(original_class_name)
            stats['classes'].setdefault(dish_name, 0)
            stats['classes'][dish_name] += 1

        # Segunda passagem: processar e salvar dados consolidados por nome original
        for original_class_name, total_area in dish_areas_by_original_name.items():
            # Nome traduzido para lógicas de negócio (área máxima, etc.)
            dish_name = self.dish_name_replacer.get_replacement(original_class_name)

            # A área máxima é consolidada com base no nome TRADUZIDO.
            area_percentage = self.update_consolidated_max_area(dish_name, camera_id, total_area)
            
            # Salva os dados consolidados se for a melhor câmera, passando o nome original
            self.save_if_best_camera(camera_id, dish_name, area_percentage, original_class_name)

            # Atualiza estatísticas
            stats['area_percentages'][f"{camera_id}_{dish_name}"] = area_percentage
            
            if area_percentage < 0.3:
                stats['needs_refill'].append({
                    'id': f"{camera_id}_{original_class_name}", 
                    'class': dish_name, 
                    'percentage': area_percentage
                })

            # Desenha todas as detecções individuais no frame
            for bbox, confidence in all_detections_by_original_name[original_class_name]:
                annotated_frame = self.draw_detection(
                    frame=annotated_frame,
                    label_text=f"{dish_name} (Total: {area_percentage:.1%})",
                    color_key=original_class_name,
                    confidence=confidence,
                    bbox=bbox
                )

        return annotated_frame, stats
    
    def draw_detection(self, frame, label_text, color_key, confidence, bbox):
        """
        Desenha uma detecção no frame com bounding box, classe e porcentagem.
        - label_text: O texto a ser exibido (ex: "Arroz Branco (Total: 85%)").
        - color_key: Não mais utilizado, todas as detecções são verdes.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Cor verde fixa para todas as detecções
        color_bgr = (255, 255, 255)
        
        # Desenha a bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, self.line_thickness)
        
        # Calcula o tamanho do texto completo
        text_size, _ = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)
        text_width, text_height = text_size
        
        # Fundo para o texto na parte inferior
        cv2.rectangle(
            frame, 
            (x1, y2 - text_height - 5), 
            (x1 + text_width, y2), 
            color_bgr, 
            -1
        )
        
        # Texto na parte inferior
        cv2.putText(
            frame, 
            label_text, 
            (x1, y2 - 5), 
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
        Agora inclui verificação de data para reset diário.
        """
        percentage = 1.0
        current_date = datetime.now().date()
        
        with self.max_areas_lock:
            # Verifica se precisa resetar as áreas máximas
            if current_date != self.current_date:
                self.logger.info("Novo dia detectado. Resetando áreas máximas...")
                self.max_areas.clear()
                self.current_date = current_date
                self.logger.info(f"Áreas máximas resetadas. Nova data de referência: {self.current_date}")
            
            if dish_name not in self.max_areas:
                self.max_areas[dish_name] = {
                    'max_area': current_area,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'reference_date': current_date
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
    
    def save_if_best_camera(self, camera_id, dish_name, percentage, original_class_name):
        """
        Salva os dados apenas se esta câmera tem a maior porcentagem para este prato.
        """
        is_best_camera = False
        with self.best_cameras_lock:
            if dish_name in self.best_cameras:
                is_best_camera = self.best_cameras[dish_name]['camera_id'] == camera_id
        
        if is_best_camera:
            self.save_area_percentage(camera_id, dish_name, percentage, original_class_name)
    
    def save_area_percentage(self, camera_id, dish_name, percentage, original_class_name):
        """
        Salva a porcentagem de área de um prato no arquivo JSON, seguindo a estrutura da imagem.
        """
        try:
            with self.data_file_lock:
                # Usar r+ para ler e depois escrever no mesmo arquivo
                with open(self.data_file, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {"address": []} # Arquivo vazio ou corrompido

                    restaurant_name = self.camera_info[camera_id]["restaurant"]
                    location_name = self.camera_info[camera_id].get("location_name", f"Local {camera_id}")
                    dish_id = f"{camera_id}_{original_class_name}"

                    # Encontra ou cria o restaurante
                    restaurant_data = next((r for r in data["address"] if r["restaurant"] == restaurant_name), None)
                    if not restaurant_data:
                        restaurant_data = {"restaurant": restaurant_name, "locations": []}
                        data["address"].append(restaurant_data)

                    # Encontra ou cria a localização
                    location_data = next((loc for loc in restaurant_data["locations"] if loc["location_id"] == camera_id), None)
                    if not location_data:
                        location_data = {
                            "location_id": camera_id,
                            "location_name": location_name,
                            "dishes": []
                        }
                        restaurant_data["locations"].append(location_data)

                    # Encontra ou cria o prato
                    dish_data = next((d for d in location_data["dishes"] if d["dish_id"] == dish_id), None)
                    if not dish_data:
                        dish_data = {
                            "dish_id": dish_id,
                            "dish_name": dish_name,
                            "percentage_remaining": 0,
                            "needs_reposition": False,
                            "timestamp": ""
                        }
                        location_data["dishes"].append(dish_data)
                    
                    # Atualiza os dados do prato
                    dish_data["percentage_remaining"] = int(percentage * 100)
                    dish_data["needs_reposition"] = "true" if percentage < 0.3 else "false" # Convertido para string
                    dish_data["timestamp"] = datetime.now().isoformat()
                    
                    # Volta ao início do arquivo para sobrescrever
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()

                if self.external_client:
                    # Envia apenas os dados relevantes para a API, e não o arquivo inteiro
                    payload = {
                        "address": [{
                            "restaurant": restaurant_name,
                            "locations": [{
                                "location_id": camera_id,
                                "location_name": location_name,
                                "dishes": [dish_data] # Apenas o prato atualizado
                            }]
                        }]
                    }
                    if self.external_client.send_data(payload):
                        self.logger.info(f"Dados enviados com sucesso para o dashboard: {restaurant_name} - {location_name} - {dish_name}: {percentage:.1%}")
                    
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados no arquivo JSON: {e}", exc_info=True) # Adicionado exc_info para mais detalhes
    
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
        Agora também verifica a data de referência.
        """
        current_time = time.time()
        current_date = datetime.now().date()
        dishes_to_remove = []
        
        with self.max_areas_lock:
            for dish_name, data in self.max_areas.items():
                last_seen = data['last_seen']
                # Remove se passou do tempo máximo ou se é de um dia anterior
                if (current_time - last_seen > max_age_seconds):
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
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        text_to_display = f"Camera: {camera_id} | {timestamp_str}" if camera_id else timestamp_str
        
        # Parâmetros de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255) # Branco
        shadow_color = (0, 0, 0) # Preto
        
        # Posição do texto (canto inferior esquerdo)
        text_size, _ = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)
        position = (10, frame.shape[0] - 10) # 10 pixels da borda inferior esquerda
        
        # Adiciona sombra para melhor legibilidade
        cv2.putText(frame, text_to_display, (position[0] + 1, position[1] + 1), font, font_scale, shadow_color, font_thickness)
        
        # Adiciona o texto principal
        cv2.putText(frame, text_to_display, position, font, font_scale, text_color, font_thickness)
        
        return frame

    def add_info_box(self, frame, info_lines):
        """
        Adiciona uma caixa de informações ao frame.
        
        Args:
            frame: O frame a ser modificado
            info_lines: Uma lista de strings, cada uma sendo uma linha de informação
        """
        if frame is None:
            return None
        
        # Parâmetros
        box_color = (0, 0, 0)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        padding = 10
        
        box_height = len(info_lines) * line_height + 2 * padding
        box_width = 300 # Largura fixa
        
        overlay = frame.copy()
        
        cv2.rectangle(
            overlay,
            (padding, padding),
            (padding + box_width, padding + box_height),
            box_color,
            -1
        )
        
        # Adiciona transparência à caixa
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_pos = padding + (i + 1) * line_height
            cv2.putText(frame, line, (padding + 5, y_pos), font, font_scale, text_color, 1)
            
        return frame 
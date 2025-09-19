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
from pathlib import Path  # noqa: F401 (compat: pode ser usado em futuras funcionalidades)
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
        # Estabilização de prioridade entre câmeras (histerese)
        self.best_camera_margin = 0.05  # requer vantagem mínima de 5 p.p. para trocar
        self.best_camera_min_hold_seconds = 5.0  # vantagem deve se manter por 5s
        self.best_camera_pending = {}  # { dish_name: { 'camera_id': str, 'start_time': float } }
        
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
        
        # Estado para filtro/atraso de mudança de porcentagem por prato
        self.percentage_state = {}  # { dish_name: { 'confirmed': float, 'pending': { 'start_time': float, 'initial': float, 'latest': float }|None } }
        self.percentage_state_lock = Lock()
        self.change_delay_seconds = 5.0  # atraso para confirmar mudança
        
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
    
    def process_detections(self, frame, camera_id, boxes, class_names, masks=None):
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

        # Dicionários para acumular por nome de prato ORIGINAL
        dish_areas_by_original_name = {}  # key: original_class_name, value: total_area
        detections_by_original_name = {}   # key: original_class_name, value: list of dicts com bbox/mask/score
        label_boxes_by_original_name = {}  # key: original_class_name, value: bbox de referência (união)


        # Primeira passagem: acumular áreas e informações de detecção por nome original
        have_masks = hasattr(masks, 'data') and masks is not None
        mask_data = None
        if have_masks:
            try:
                mask_data = masks.data.cpu().numpy()  # (N, H, W) com valores 0..1
            except Exception:
                mask_data = None
                have_masks = False

        detected_original_names = set()
        detected_translated_names = set()
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            confidence = boxes.conf[i].item()
            class_id = int(boxes.cls[i].item())
            original_class_name = class_names[class_id]
            detected_original_names.add(original_class_name)

            # Área via máscara se disponível, senão bbox (legado)
            mask_area = 0
            instance_mask = None
            if have_masks and mask_data is not None and i < len(mask_data):
                instance_mask = (mask_data[i] >= 0.5).astype(np.uint8)  # binária
                mask_area = int(instance_mask.sum())

            area = mask_area if mask_area > 0 else self.calculate_area(bbox)

            dish_areas_by_original_name.setdefault(original_class_name, 0)
            dish_areas_by_original_name[original_class_name] += area

            detections_by_original_name.setdefault(original_class_name, [])
            detections_by_original_name[original_class_name].append({
                'bbox': bbox,
                'confidence': confidence,
                'mask': instance_mask
            })

            # Atualiza caixa de rótulo (união de bboxes por prato)
            if original_class_name not in label_boxes_by_original_name:
                label_boxes_by_original_name[original_class_name] = bbox.copy()
            else:
                x1, y1, x2, y2 = label_boxes_by_original_name[original_class_name]
                nx1, ny1, nx2, ny2 = bbox
                label_boxes_by_original_name[original_class_name] = np.array([
                    min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)
                ])

            # Estatísticas por nome traduzido
            dish_name = self.dish_name_replacer.get_replacement(original_class_name)
            detected_translated_names.add(dish_name)
            stats['classes'].setdefault(dish_name, 0)
            stats['classes'][dish_name] += 1

        # Segunda passagem: processar e salvar dados consolidados por nome original
        for original_class_name, total_area in dish_areas_by_original_name.items():
            # Nome traduzido para lógicas de negócio (área máxima, etc.)
            dish_name = self.dish_name_replacer.get_replacement(original_class_name)

            # Porcentagem instantânea por câmera (sem filtro), baseada na área máxima configurada
            configured_max = self.dish_name_replacer.get_max_area_by_translated(dish_name)
            if isinstance(configured_max, (int, float)) and configured_max > 0:
                instant_percentage = min(1.0, float(total_area) / float(configured_max))
            else:
                # Fallback dinâmico simples: usa max atual se existir
                with self.max_areas_lock:
                    current_max = self.max_areas.get(dish_name, {}).get('max_area', 0)
                instant_percentage = 1.0 if current_max <= 0 else float(total_area) / float(current_max)
                instant_percentage = max(0.0, min(1.0, instant_percentage))

            # A área máxima consolidada/histerese continua sendo atualizada para persistência/"melhor câmera"
            area_percentage_filtered = self.update_consolidated_max_area(dish_name, camera_id, total_area)
            
            # Salva os dados consolidados se for a melhor câmera, passando o nome original
            self.save_if_best_camera(camera_id, dish_name, area_percentage_filtered, original_class_name)

            # Atualiza estatísticas com o valor filtrado (para persistência entre frames)
            stats['area_percentages'][f"{camera_id}_{dish_name}"] = area_percentage_filtered
            
            if area_percentage_filtered < 0.3:
                stats['needs_refill'].append({
                    'id': f"{camera_id}_{original_class_name}", 
                    'class': dish_name, 
                    'percentage': area_percentage_filtered
                })

            # Desenha overlays das máscaras/caixas por detecção
            for det in detections_by_original_name.get(original_class_name, []):
                if det['mask'] is not None:
                    annotated_frame = self.draw_mask_overlay(annotated_frame, det['mask'])
                else:
                    # Fallback: desenha bbox leve
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Desenha UM rótulo por prato, no canto superior da bbox unificada
            if original_class_name in label_boxes_by_original_name:
                x1, y1, x2, y2 = map(int, label_boxes_by_original_name[original_class_name])
                area_display = self._format_area(total_area)
                label_text = f"{dish_name} ({instant_percentage:.0%} | {area_display})"
                annotated_frame = self.draw_label_box(
                    annotated_frame,
                    x1,
                    y1,
                    label_text
                )

        # Após processar os detectados, enviar 0% para pratos esperados (já conhecidos) ausentes
        try:
            absent_original_names = self._get_absent_known_originals(camera_id, detected_original_names)
            for original_class_name in absent_original_names:
                dish_name = self.dish_name_replacer.get_replacement(original_class_name)
                # Se algum alias deste prato foi detectado, não forçar 0%
                if dish_name in detected_translated_names:
                    continue
                # Só atualiza se já temos calibração (evita criar max_area=0)
                with self.max_areas_lock:
                    has_calibration = dish_name in self.max_areas and self.max_areas[dish_name].get('max_area', 0) > 0
                if not has_calibration:
                    continue

                area_percentage = self.update_consolidated_max_area(dish_name, camera_id, 0, force_immediate=True)
                # Grava se esta câmera for a melhor para o prato (pode ou não gravar, conforme regra global)
                self.save_if_best_camera(camera_id, dish_name, area_percentage, original_class_name)
                stats['area_percentages'][f"{camera_id}_{dish_name}"] = area_percentage
                if area_percentage < 0.3:
                    stats['needs_refill'].append({
                        'id': f"{camera_id}_{original_class_name}",
                        'class': dish_name,
                        'percentage': area_percentage
                    })
        except Exception as e:
            self.logger.error(f"Falha ao processar ausentes 0%: {e}")

        return annotated_frame, stats

    def _get_absent_known_originals(self, camera_id, detected_original_names):
        """
        Retorna o conjunto de nomes originais (do modelo) já conhecidos para a câmera
        que não apareceram nas detecções atuais.
        Baseado no arquivo JSON atual (ou seja, pratos já vistos/registrados anteriormente).
        """
        known_originals = set()
        try:
            data = self.load_area_data(camera_id=camera_id)
            for restaurant in data.get('address', []):
                for location in restaurant.get('locations', []):
                    if location.get('location_id') != camera_id:
                        continue
                    for dish in location.get('dishes', []):
                        dish_id = dish.get('dish_id', '')
                        prefix = f"{camera_id}_"
                        if dish_id.startswith(prefix):
                            original = dish_id[len(prefix):]
                            if original:
                                known_originals.add(original)
        except Exception as e:
            self.logger.error(f"Erro obtendo pratos conhecidos para {camera_id}: {e}")
            return set()

        # Ausentes = conhecidos - detectados
        return known_originals.difference(detected_original_names)

        
        
    
    def draw_detection(self, frame, label_text, color_key, confidence, bbox):
        # Método legado: manter compatibilidade chamando draw_label_box no topo da bbox
        x1, y1, x2, y2 = map(int, bbox)
        return self.draw_label_box(frame, x1, y1, label_text)

    def draw_mask_overlay(self, frame, mask_binary):
        """
        Desenha overlay semi-transparente e contorno de uma máscara binária no frame.
        """
        if mask_binary is None:
            return frame

        h, w = frame.shape[:2]
        if mask_binary.shape != (h, w):
            # Ajusta máscara se necessário
            mask_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_binary

        overlay = frame.copy()
        color = (0, 255, 0)  # verde
        # Aplica cor onde mask==1
        colored = np.zeros_like(frame)
        colored[:, :] = color
        overlay = np.where(mask_resized[:, :, None] == 1, colored, overlay)
        # Alpha blend 50%
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Contorno
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 200, 0), 2)
        return frame

    def draw_label_box(self, frame, x, y, text):
        """
        Desenha uma pequena caixa de rótulo no canto superior da região indicada.
        """
        color_bgr = (0, 255, 0)
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        text_width, text_height = text_size
        x2 = x + text_width + 6
        y2 = y + text_height + 8
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        cv2.rectangle(frame, (x, y), (x2, y2), color_bgr, -1)
        cv2.putText(frame, text, (x + 3, y + text_height + 1), self.font, self.font_scale, (0, 0, 0), self.font_thickness)
        return frame

    def _format_area(self, area_pixels):
        """
        Formata a área em pixels para exibição compacta (ex.: 1.5k px, 2.0M px).
        """
        try:
            area = float(area_pixels)
        except Exception:
            area = 0.0
        if area >= 1_000_000:
            return f"{area/1_000_000:.1f}M px"
        if area >= 1_000:
            return f"{area/1_000:.1f}k px"
        return f"{int(area)} px"
    
    def calculate_area(self, bbox):
        """
        Calcula a área de um bounding box.
        """
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height
    
    def update_consolidated_max_area(self, dish_name, camera_id, current_area, force_immediate=False):
        """
        Atualiza a área máxima consolidada para um prato e retorna a porcentagem atual.
        Agora inclui verificação de data para reset diário.
        Quando existir área máxima configurada no JSON, usa-a diretamente
        (porcentagem = min(1.0, area_atual / area_max_configurada)).
        """
        raw_percentage = 1.0
        current_date = datetime.now().date()
        
        reset_percent_state = False
        with self.max_areas_lock:
            # Verifica se precisa resetar as áreas máximas
            if current_date != self.current_date:
                self.logger.info("Novo dia detectado. Resetando áreas máximas...")
                self.max_areas.clear()
                self.current_date = current_date
                self.logger.info(f"Áreas máximas resetadas. Nova data de referência: {self.current_date}")
                reset_percent_state = True
            
            # Consulta área máxima configurada (por nome traduzido)
            configured_max = self.dish_name_replacer.get_max_area_by_translated(dish_name)
            if isinstance(configured_max, (int, float)) and configured_max > 0:
                # Usa área máxima fixa e satura em 100%
                raw_percentage = min(1.0, float(current_area) / float(configured_max))
            else:
                # Fallback para comportamento antigo (auto-calibração)
                if dish_name not in self.max_areas:
                    self.max_areas[dish_name] = {
                        'max_area': current_area,
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'reference_date': current_date
                    }
                    raw_percentage = 1.0
                    self.logger.debug(f"Novo prato registrado: {dish_name} (área: {current_area})")
                else:
                    self.max_areas[dish_name]['last_seen'] = time.time()
                    current_max = self.max_areas[dish_name]['max_area']
                    
                    if current_area > current_max:
                        self.max_areas[dish_name]['max_area'] = current_area
                        raw_percentage = 1.0
                        self.logger.debug(f"Nova área máxima global para {dish_name}: {current_area} (câmera {camera_id})")
                    else:
                        raw_percentage = current_area / current_max if current_max > 0 else 1.0
        
        # Se houve reset diário, limpar também o estado de porcentagens confirmadas
        if reset_percent_state:
            with self.percentage_state_lock:
                self.percentage_state.clear()

        # Aplicar filtro de atraso/limiar na porcentagem
        filtered_percentage = self._filter_percentage(dish_name, raw_percentage, force_immediate=force_immediate)
        
        with self.best_cameras_lock:
            now_ts = time.time()
            current_best = self.best_cameras.get(dish_name)
            if current_best is None:
                # Primeira câmera observada para o prato vira a melhor
                self.best_cameras[dish_name] = {
                    'camera_id': camera_id,
                    'percentage': filtered_percentage,
                    'timestamp': now_ts
                }
                # limpa pendência
                if dish_name in self.best_camera_pending:
                    del self.best_camera_pending[dish_name]
            else:
                if camera_id == current_best.get('camera_id'):
                    # Atualiza métricas da melhor atual e limpa pendência
                    current_best['percentage'] = filtered_percentage
                    current_best['timestamp'] = now_ts
                    if dish_name in self.best_camera_pending:
                        del self.best_camera_pending[dish_name]
                else:
                    # Desafiadora: verifica vantagem e janela de hold
                    advantage = filtered_percentage - float(current_best.get('percentage', 0.0))
                    pending = self.best_camera_pending.get(dish_name)
                    if advantage >= self.best_camera_margin:
                        if not pending or pending.get('camera_id') != camera_id:
                            # inicia pendência para esta câmera
                            self.best_camera_pending[dish_name] = {
                                'camera_id': camera_id,
                                'start_time': now_ts
                            }
                        else:
                            # já pendente; verifica se manteve vantagem por tempo suficiente
                            elapsed = now_ts - float(pending.get('start_time', now_ts))
                            if elapsed >= self.best_camera_min_hold_seconds:
                                # troca a melhor câmera
                                self.best_cameras[dish_name] = {
                                    'camera_id': camera_id,
                                    'percentage': filtered_percentage,
                                    'timestamp': now_ts
                                }
                                del self.best_camera_pending[dish_name]
                    else:
                        # vantagem insuficiente: cancela pendência se for deste desafiante
                        if pending and pending.get('camera_id') == camera_id:
                            del self.best_camera_pending[dish_name]
        
        return filtered_percentage
    
    def _filter_percentage(self, dish_name, raw_percentage, force_immediate=False):
        """
        Aplica apenas atraso de confirmação de 5s nas mudanças de porcentagem.
        Retorna a porcentagem "confirmada" a ser exibida/salva.
        """
        now_ts = time.time()
        with self.percentage_state_lock:
            state = self.percentage_state.get(dish_name)
            if state is None:
                # Primeiro valor observado: confirma imediatamente
                self.percentage_state[dish_name] = {
                    'confirmed': float(raw_percentage),
                    'pending': None
                }
                return float(raw_percentage)

            confirmed = state.get('confirmed', 1.0)
            pending = state.get('pending')

            # Força confirmação imediata (ex.: ausência -> 0%)
            if force_immediate:
                state['confirmed'] = float(raw_percentage)
                state['pending'] = None
                return float(state['confirmed'])

            # Se não há pendência e houve qualquer mudança, inicia janela de confirmação
            if not pending:
                if float(raw_percentage) != float(confirmed):
                    state['pending'] = {
                        'start_time': now_ts,
                        'initial': float(raw_percentage),
                        'latest': float(raw_percentage)
                    }
                    self.logger.debug(f"Mudança pendente iniciada para {dish_name}: de {confirmed:.3f} -> {raw_percentage:.3f}")
                # Enquanto não confirmar, mantém confirmado
                return float(state['confirmed'])

            # Atualiza último valor observado durante a janela
            pending['latest'] = float(raw_percentage)

            elapsed = now_ts - pending['start_time']
            if elapsed >= self.change_delay_seconds:
                previous_confirmed = state['confirmed']
                # Confirma a mudança para o último valor observado na janela
                state['confirmed'] = float(pending['latest'])
                self.logger.debug(
                    f"Mudança confirmada para {dish_name} após {elapsed:.1f}s: {previous_confirmed:.3f} -> {state['confirmed']:.3f}"
                )
                state['pending'] = None

            return float(state['confirmed'])
    
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
                        self.logger.debug(f"Dados enviados com sucesso para o dashboard: {restaurant_name} - {location_name} - {dish_name}: {percentage:.1%}")
                    
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
        
        # Limpa estado do filtro para os pratos removidos
        if dishes_to_remove:
            with self.percentage_state_lock:
                for dish_name in dishes_to_remove:
                    if dish_name in self.percentage_state:
                        del self.percentage_state[dish_name]
                
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

    def draw_persistent_boxes(self, frame, boxes, class_names, camera_id, last_percentages):
        """
        Desenha caixas de detecção persistentes em um frame sem recalcular ou salvar dados.
        Exibe a última porcentagem conhecida para cada prato.
        """
        annotated_frame = frame.copy()
        # Desenha apenas UM rótulo por prato, usando união de caixas
        label_boxes_by_name = {}
        areas_by_name = {}
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            class_id = int(boxes.cls[i].item())
            original_class_name = class_names[class_id]
            dish_name = self.dish_name_replacer.get_replacement(original_class_name)

            if dish_name not in label_boxes_by_name:
                label_boxes_by_name[dish_name] = bbox.copy()
            else:
                x1, y1, x2, y2 = label_boxes_by_name[dish_name]
                nx1, ny1, nx2, ny2 = bbox
                label_boxes_by_name[dish_name] = np.array([
                    min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)
                ])
            # Acumula área por prato (via bbox)
            areas_by_name[dish_name] = areas_by_name.get(dish_name, 0) + self.calculate_area(bbox)

        for dish_name, bbox in label_boxes_by_name.items():
            percentage = last_percentages.get(dish_name, 0.0)
            area_display = self._format_area(areas_by_name.get(dish_name, 0))
            label_text = f"{dish_name} ({percentage:.0%} | {area_display})"
            x1, y1, x2, y2 = map(int, bbox)
            annotated_frame = self.draw_label_box(annotated_frame, x1, y1, label_text)

        return annotated_frame

    def draw_persistent_masks(self, frame, boxes, masks, class_names, camera_id, last_percentages):
        """
        Desenha overlays de máscaras e UM rótulo por prato usando últimas porcentagens.
        """
        annotated_frame = frame.copy()
        have_masks = hasattr(masks, 'data') and masks is not None
        mask_data = None
        if have_masks:
            try:
                mask_data = masks.data.cpu().numpy()
            except Exception:
                mask_data = None
                have_masks = False

        label_boxes_by_name = {}
        areas_by_name = {}
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            class_id = int(boxes.cls[i].item())
            original_class_name = class_names[class_id]
            dish_name = self.dish_name_replacer.get_replacement(original_class_name)

            if have_masks and mask_data is not None and i < len(mask_data):
                instance_mask = (mask_data[i] >= 0.5).astype(np.uint8)
                annotated_frame = self.draw_mask_overlay(annotated_frame, instance_mask)
                areas_by_name[dish_name] = areas_by_name.get(dish_name, 0) + int(instance_mask.sum())
            else:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                areas_by_name[dish_name] = areas_by_name.get(dish_name, 0) + self.calculate_area(bbox)

            if dish_name not in label_boxes_by_name:
                label_boxes_by_name[dish_name] = bbox.copy()
            else:
                x1, y1, x2, y2 = label_boxes_by_name[dish_name]
                nx1, ny1, nx2, ny2 = bbox
                label_boxes_by_name[dish_name] = np.array([
                    min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)
                ])

        for dish_name, bbox in label_boxes_by_name.items():
            percentage = last_percentages.get(dish_name, 0.0)
            area_display = self._format_area(areas_by_name.get(dish_name, 0))
            label_text = f"{dish_name} ({percentage:.0%} | {area_display})"
            x1, y1, x2, y2 = map(int, bbox)
            annotated_frame = self.draw_label_box(annotated_frame, x1, y1, label_text)

        return annotated_frame

    def reset_daily_data(self):
        """
        Apaga o arquivo de dados JSON e re-inicializa o estado interno.
        Ideal para ser chamado diariamente para começar um novo registro.
        """
        self.logger.info("Executando limpeza diária dos dados de detecção.")
        
        with self.data_file_lock:
            # Apaga o arquivo de dados se ele existir
            if os.path.exists(self.data_file):
                try:
                    os.remove(self.data_file)
                    self.logger.info(f"Arquivo de dados '{self.data_file}' apagado com sucesso.")
                except OSError as e:
                    self.logger.error(f"Erro ao apagar o arquivo de dados '{self.data_file}': {e}")
            
            # Re-inicializa o cache de áreas máximas
            self.max_areas = {}
            self.logger.info("Cache de áreas máximas foi re-inicializado.")

            # Re-inicializa estado filtrado de porcentagens
            with self.percentage_state_lock:
                self.percentage_state = {}

            # Cria um arquivo vazio para garantir que ele exista para o próximo ciclo
            try:
                with open(self.data_file, 'w') as f:
                    json.dump({}, f)
                self.logger.info(f"Arquivo de dados '{self.data_file}' recriado vazio.")
            except IOError as e:
                self.logger.error(f"Não foi possível recriar o arquivo de dados '{self.data_file}': {e}")


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
        
    def create_status_frame(self, text, width=800, height=600):
        """
        Cria um frame preto com uma mensagem de status centralizada.
        
        Args:
            text (str): A mensagem a ser exibida.
            width (int): A largura do frame.
            height (int): A altura do frame.
            
        Returns:
            np.ndarray: Um frame com a mensagem de status.
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        return frame
        
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
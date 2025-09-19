#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Mapeamento de Nomes de Pratos.
Responsável por traduzir os nomes de classes genéricas do modelo (ex: 'food_tray')
para nomes de pratos específicos (ex: 'Arroz Branco') definidos em um arquivo JSON.
Agora também suporta área máxima por prato e câmera primária.
"""

import json
import logging
from pathlib import Path
import threading

class DishNameReplacer:
    """
    Classe para carregar e fornecer traduções de nomes de pratos e áreas máximas.
    Suporta dois formatos de JSON:
    - formato antigo: { "Atum": "atum_selado", ... }
    - formato novo:   { "Atum": { "name": "atum_selado", "max_area_px": 46200, "primary_camera_id": "cam02" }, ... }
    """
    
    def __init__(self, names_file="config/nomes.json"):
        self.logger = logging.getLogger(__name__)
        self.names_file = Path(names_file)
        self.original_to_translated = {}
        self.translated_to_max_area = {}
        self.translated_to_primary_camera = {}
        self.names_lock = threading.Lock()
        
        self._load_dish_names()
        self.logger.debug("DishNameReplacer inicializado.")
    
    def _load_dish_names(self):
        try:
            if not self.names_file.exists():
                self.logger.warning(f"Arquivo de nomes não encontrado em '{self.names_file}'. O sistema usará os nomes de classe originais.")
                return

            with self.names_lock:
                with open(self.names_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.original_to_translated = {}
                self.translated_to_max_area = {}
                self.translated_to_primary_camera = {}

                for original, value in raw.items():
                    if isinstance(value, dict):
                        translated = value.get("name", original)
                        max_area = value.get("max_area_px")
                        primary_cam = value.get("primary_camera_id")
                        self.original_to_translated[original] = translated
                        if isinstance(max_area, (int, float)) and max_area > 0:
                            self.translated_to_max_area[translated] = float(max_area)
                        if isinstance(primary_cam, str) and primary_cam:
                            self.translated_to_primary_camera[translated] = primary_cam
                    else:
                        translated = str(value)
                        self.original_to_translated[original] = translated
                self.logger.info(
                    f"Carregados {len(self.original_to_translated)} pratos (max:{len(self.translated_to_max_area)}, prim:{len(self.translated_to_primary_camera)}) de {self.names_file}"
                )
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar o arquivo JSON '{self.names_file}': {e}")
        except Exception as e:
            self.logger.error(f"Erro inesperado ao carregar nomes de pratos: {e}")

    def get_replacement(self, original_name):
        with self.names_lock:
            return self.original_to_translated.get(original_name, original_name)

    def get_max_area_by_translated(self, translated_name):
        with self.names_lock:
            return self.translated_to_max_area.get(translated_name)

    def get_primary_camera_by_translated(self, translated_name):
        """Retorna o camera_id primário para o prato traduzido, se houver."""
        with self.names_lock:
            return self.translated_to_primary_camera.get(translated_name)

    def get_all_original_names(self):
        with self.names_lock:
            return list(self.original_to_translated.keys())
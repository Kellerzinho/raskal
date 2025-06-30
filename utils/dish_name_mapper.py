#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Mapeamento de Nomes de Pratos.
Responsável por traduzir os nomes de classes genéricas do modelo (ex: 'food_tray')
para nomes de pratos específicos (ex: 'Arroz Branco') definidos em um arquivo JSON.
"""

import json
import logging
from pathlib import Path
import threading

class DishNameReplacer:
    """
    Classe para carregar e fornecer traduções de nomes de pratos.
    """
    
    def __init__(self, names_file="config/nomes.json"):
        """
        Inicializa o mapeador de nomes.
        
        Args:
            names_file: Caminho para o arquivo JSON com os nomes dos pratos.
        """
        self.logger = logging.getLogger(__name__)
        self.names_file = Path(names_file)
        self.dish_names = {}
        self.names_lock = threading.Lock()
        
        self._load_dish_names()
        self.logger.debug("DishNameReplacer inicializado.")
    
    def _load_dish_names(self):
        """
        Carrega os nomes dos pratos do arquivo JSON.
        Se o arquivo não existir, opera com um dicionário vazio.
        """
        try:
            if self.names_file.exists():
                with self.names_lock:
                    with open(self.names_file, 'r', encoding='utf-8') as f:
                        self.dish_names = json.load(f)
                    self.logger.info(f"Carregados {len(self.dish_names)} nomes de pratos de {self.names_file}")
            else:
                self.logger.warning(f"Arquivo de nomes não encontrado em '{self.names_file}'. O sistema usará os nomes de classe originais.")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar o arquivo JSON '{self.names_file}': {e}")
        except Exception as e:
            self.logger.error(f"Erro inesperado ao carregar nomes de pratos: {e}")

    def get_replacement(self, original_name):
        """
        Obtém o nome traduzido para uma classe.
        Se a tradução não for encontrada, retorna o nome original.

        Args:
            original_name (str): O nome da classe detectado pelo modelo.

        Returns:
            str: O nome traduzido ou o nome original.
        """
        with self.names_lock:
            # Retorna o nome do dicionário, ou a 'original_name' como padrão.
            return self.dish_names.get(original_name, original_name)
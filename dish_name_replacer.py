#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Substituição de Nomes de Pratos - Sistema de Monitoramento de Buffet
Responsável por substituir os códigos de pratos pelos nomes reais definidos em nomes.json
durante o processo de salvamento dos dados no buffet_data.json.
"""

import json
import os
import logging
from pathlib import Path
import threading
from functools import wraps
import inspect
import copy  # CORREÇÃO: Importado para deepcopy


class DishNameReplacer:
    """
    Classe para substituir nomes de pratos no momento do salvamento em buffet_data.json
    pelos nomes reais definidos em nomes.json.
    """
    
    def __init__(self, names_file="config/nomes.json"):
        """
        Inicializa o processador de substituição de nomes.
        
        Args:
            names_file: Caminho para o arquivo JSON com os nomes reais dos pratos
        """
        self.logger = logging.getLogger(__name__)
        self.names_file = Path(names_file)
        self.dish_names = {}
        self.names_lock = threading.Lock()
        self.last_loaded_time = 0
        self.reload_interval = 300  # Recarregar a cada 5 minutos
        
        # Carregar nomes de pratos inicialmente
        self._load_dish_names()
        
        self.logger.debug("DishNameReplacer inicializado")
    
    def _load_dish_names(self):
        """
        Carrega os nomes dos pratos do arquivo JSON.
        Verifica se o arquivo existe e pode ser lido.
        """
        try:
            if not self.names_file.exists():
                self.logger.warning(f"Arquivo de nomes não encontrado: {self.names_file}")
                # Tentar procurar em caminhos alternativos
                alternative_paths = [
                    Path("nomes.json"),
                    Path("../config/nomes.json"),
                    Path("./config/nomes.json")
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        self.names_file = alt_path
                        self.logger.info(f"Encontrado arquivo de nomes em caminho alternativo: {alt_path}")
                        break
                else:
                    return  # Nenhum arquivo encontrado
                
            with self.names_lock:
                with open(self.names_file, 'r', encoding='utf-8') as f:
                    self.dish_names = json.load(f)
                self.last_loaded_time = os.path.getmtime(self.names_file)
                
            self.logger.info(f"Carregados {len(self.dish_names)} nomes de pratos do arquivo {self.names_file}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar arquivo JSON {self.names_file}: {e}")
        except Exception as e:
            self.logger.error(f"Erro ao carregar nomes de pratos: {e}")
    
    def _check_reload_names(self):
        """
        Verifica se é necessário recarregar os nomes de pratos.
        Recarrega se o arquivo foi modificado desde o último carregamento.
        """
        try:
            if not self.names_file.exists():
                return
                
            current_mtime = os.path.getmtime(self.names_file)
            
            # Se o arquivo foi modificado, recarregar
            if current_mtime > self.last_loaded_time:
                self.logger.info(f"Arquivo de nomes modificado, recarregando...")
                self._load_dish_names()
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar modificação do arquivo de nomes: {e}")
    
    def replace_dish_names_in_data(self, data):
        """
        Substitui os nomes de pratos no objeto de dados antes do salvamento.
        
        Args:
            data: Estrutura de dados com informações do buffet
            
        Returns:
            dict: Estrutura de dados com nomes substituídos
        """
        # Verificar se é necessário recarregar os nomes
        self._check_reload_names()
        
        # Se não temos nomes carregados, retornar dados originais
        if not self.dish_names:
            return data
        
        # Clonar os dados para não modificar o objeto original
        updated_data = self._deep_copy(data)
        
        # Contar substituições
        replaced_count = 0
        
        # Percorrer a estrutura de dados do buffet
        for restaurant in updated_data.get("address", []):
            for location in restaurant.get("locations", []):
                for dish in location.get("dishes", []):
                    # Verificar se o código do prato está no dicionário de nomes
                    dish_code = dish.get("dish_name")
                    if dish_code in self.dish_names:
                        # Substituir pelo nome real
                        dish["dish_name"] = self.dish_names[dish_code]
                        replaced_count += 1
        
        if replaced_count > 0:
            self.logger.debug(f"Substituídos {replaced_count} nomes de pratos")
        
        return updated_data
    
    def _deep_copy(self, obj):
        """
        Cria uma cópia profunda de um objeto.
        
        Args:
            obj: Objeto a ser copiado
            
        Returns:
            Cópia profunda do objeto
        """
        # CORREÇÃO: Utilizando copy.deepcopy por ser mais idiomático e eficiente
        # para objetos Python em geral, em vez de depender da serialização JSON.
        return copy.deepcopy(obj)


# Função para monkey-patch o método de salvamento da classe DetectionProcessor
def patch_detection_processor():
    """
    Aplica patch na classe DetectionProcessor para substituir os nomes de pratos
    durante o salvamento dos dados.
    """
    try:
        # Importar as classes necessárias
        from processing import DetectionProcessor
        
        # Criar o replacer
        replacer = DishNameReplacer()
        
        # Definir a função de wrapper para save_area_percentage
        def save_area_percentage_wrapper(original_method):
            @wraps(original_method)
            def wrapper(self, camera_id, dish_id, dish_name, percentage):
                # Se o dish_name estiver no dicionário de nomes, usá-lo para exibição
                if dish_name in replacer.dish_names:
                    real_name = replacer.dish_names[dish_name]
                    return original_method(self, camera_id, dish_id, real_name, percentage)
                else:
                    return original_method(self, camera_id, dish_id, dish_name, percentage)
            return wrapper
        
        # Aplicar o patch ao método save_area_percentage
        if hasattr(DetectionProcessor, 'save_area_percentage'):
            original_method = DetectionProcessor.save_area_percentage
            DetectionProcessor.save_area_percentage = save_area_percentage_wrapper(original_method)
            logging.getLogger(__name__).info("Patch aplicado ao método save_area_percentage da classe DetectionProcessor")
            return True
        else:
            logging.getLogger(__name__).warning("Método save_area_percentage não encontrado na classe DetectionProcessor")
            return False
            
    except ImportError as e:
        logging.getLogger(__name__).error(f"Não foi possível importar a classe DetectionProcessor: {e}")
        return False
    except Exception as e:
        logging.getLogger(__name__).error(f"Erro ao aplicar patch: {e}")
        return False


# Função para instalar o monkey-patch quando o módulo é importado
def install():
    """
    Instala o monkey-patch para a substituição de nomes.
    """
    # Configurar logging se ainda não estiver configurado
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    logger = logging.getLogger(__name__)
    logger.info("Instalando substituição automática de nomes de pratos...")
    
    success = patch_detection_processor()
    
    if success:
        logger.info("Substituição automática de nomes de pratos instalada com sucesso")
    else:
        logger.warning("Não foi possível instalar a substituição automática de nomes de pratos")
    
    return success


# Instalar o patch automaticamente quando o módulo é importado
if __name__ != "__main__":
    install()


# Código para teste direto do módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("dish_name_replacer")
    logger.info("Teste do módulo de substituição de nomes de pratos")
    
    try:
        # Criar instância do substituidor
        replacer = DishNameReplacer()
        
        # Testar com um objeto de dados de exemplo
        test_data = {
            "address": [
                {
                    "restaurant": "rcna",
                    "locations": [
                        {
                            "location_id": "cam3",
                            "location_name": "Buffet de Saladas",
                            "dishes": [
                                {
                                    "dish_id": "cam3_cam4PratoS",
                                    "dish_name": "cam4PratoS",
                                    "percentage_remaining": 67,
                                    "needs_reposition": False
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Substituir nomes
        updated_data = replacer.replace_dish_names_in_data(test_data)
        
        # Exibir resultado
        logger.info("Dados originais:")
        logger.info(json.dumps(test_data, indent=2))
        
        logger.info("Dados com nomes substituídos:")
        logger.info(json.dumps(updated_data, indent=2))
        
        # Tentar instalar o patch
        success = install()
        logger.info(f"Instalação do patch: {'Sucesso' if success else 'Falha'}")
            
    except Exception as e:
        logger.exception(f"Erro ao testar módulo: {e}")
    
    logger.info("Teste finalizado")
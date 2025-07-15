#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo Principal - Ponto de Entrada da Aplicação
"""

import logging
import sys
from utils.logger_config import LoggerManager
from monitoring_system import BuffetMonitoringSystem

def main():
    """
    Função principal que inicializa e executa o sistema.
    """
    # Configura o sistema de logging para o nível INFO no console
    LoggerManager().setup_root_logger(console_level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Aplicação Buffet Monitor iniciada")
    
    try:
        # Instancia e inicia o sistema de monitoramento
        # O argumento show_visualization pode ser controlado por CLI no futuro
        system = BuffetMonitoringSystem(show_visualization=True)
        system.start()
        
    except Exception as e:
        logger.critical(f"Erro fatal não tratado na aplicação: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("Aplicação Buffet Monitor encerrada")

if __name__ == "__main__":
    main()
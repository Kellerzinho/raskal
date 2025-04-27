#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Monitoramento - Sistema de Monitoramento de Buffet
Responsável por acompanhar o status do sistema, incluindo câmeras,
APIs e threads de comunicação.
"""

import logging
import time
import threading
import os
import subprocess
import sys
import shutil
from datetime import datetime


class SystemMonitor:
    """
    Classe para monitoramento do status do Sistema de Monitoramento de Buffet.
    """
    
    def __init__(self, camera_config):
        """
        Inicializa o monitor de sistema com as variáveis de status.
        
        Args:
            camera_config: Configuração das câmeras (do arquivo cameras.json)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando variáveis de monitoramento do sistema")
        
        # Status da API
        self.api_started = False  # Indica se a API para receber comandos do dashboard foi iniciada
        
        # Status da comunicação com o dashboard
        self.dashboard_communication_working = False  # Indica se a comunicação com o dashboard está funcionando
        
        # Status das câmeras
        self.camera_status = {}
        
        # Inicializar status para cada câmera do config
        for camera in camera_config["cameras"]:
            camera_id = camera["id"]
            self.camera_status[camera_id] = {
                "connection_tested": False,      # Indica se o teste de conexão foi bem-sucedido
                "connection_established": False,  # Indica se a conexão está estabelecida
                "model_processing": False         # Indica se o processamento pelo modelo FVBM está ocorrendo
            }
            
        self.logger.debug(f"Variáveis de monitoramento inicializadas para {len(self.camera_status)} câmeras")


class StatusTerminal:
    """
    Classe para exibir o status do sistema em um terminal separado,
    atualizando as informações in-place em vez de adicionar novas linhas.
    """
    
    def __init__(self, system_monitor):
        """
        Inicializa o terminal de status.
        
        Args:
            system_monitor: Instância do SystemMonitor para obter os dados de status
        """
        self.logger = logging.getLogger(__name__)
        self.system_monitor = system_monitor
        self.running = False
        self.update_thread = None
        self.update_interval = 1.0  # Segundos entre atualizações
        
        # Verificar se o terminal suporta limpeza de tela
        self.terminal_width = shutil.get_terminal_size().columns
        
    def start(self):
        """
        Inicia o terminal de status em uma thread separada.
        """
        if self.running:
            self.logger.warning("Terminal de status já está em execução")
            return
        
        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="StatusTerminalThread"
        )
        self.update_thread.start()
        self.logger.info("Terminal de status iniciado")
        
    def stop(self):
        """
        Para o terminal de status.
        """
        if not self.running:
            self.logger.warning("Terminal de status não está em execução")
            return
        
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.logger.info("Terminal de status finalizado")
            
    def _clear_screen(self):
        """
        Limpa a tela do terminal.
        """
        # Usar comando apropriado dependendo do sistema operacional
        if os.name == "nt":  # Windows
            os.system("cls")
        else:  # Unix/Linux/Mac
            os.system("clear")
            
    def _update_loop(self):
        """
        Loop principal para atualização do terminal de status.
        """
        try:
            # Criar um terminal separado nos sistemas suportados
            if os.name == "nt":  # Windows
                subprocess.Popen("start cmd /c python -m monitoring_display", shell=True)
                self.logger.info("Terminal de monitoramento aberto em nova janela (Windows)")
                # Como temos um terminal separado no Windows, podemos encerrar esta thread
                self.running = False
                return
            else:
                # Em sistemas Unix, tentamos abrir um novo terminal
                try:
                    # Tentar diferentes terminais comuns em Unix
                    for term_cmd in ["gnome-terminal", "xterm", "konsole", "terminal"]:
                        try:
                            subprocess.Popen([term_cmd, "-e", f"python -m monitoring_display"])
                            self.logger.info(f"Terminal de monitoramento aberto em nova janela ({term_cmd})")
                            # Como temos um terminal separado, podemos encerrar esta thread
                            self.running = False
                            return
                        except FileNotFoundError:
                            continue
                except Exception as e:
                    self.logger.warning(f"Não foi possível abrir um terminal separado: {e}")
                    self.logger.info("Usando terminal atual para exibir status")
            
            # Se não conseguiu abrir um terminal separado, usar o atual
            while self.running:
                self._clear_screen()
                self._print_status()
                time.sleep(self.update_interval)
                
        except Exception as e:
            self.logger.exception(f"Erro no loop de atualização do terminal: {e}")
            
    def _print_status(self):
        """
        Imprime o status atual do sistema no terminal.
        """
        # Obter hora atual
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Imprimir cabeçalho
        print("=" * self.terminal_width)
        print(f"SISTEMA DE MONITORAMENTO DE BUFFET - Status em {current_time}")
        print("=" * self.terminal_width)
        print()
        
        # Status da API e Dashboard
        api_status = "ATIVA" if self.system_monitor.api_started else "INATIVA"
        dashboard_status = "FUNCIONANDO" if self.system_monitor.dashboard_communication_working else "COM PROBLEMAS"
        
        print(f"API para dashboard: {api_status}")
        print(f"Comunicação com dashboard: {dashboard_status}")
        print()
        
        # Status das câmeras
        print("STATUS DAS CÂMERAS:")
        print("=" * self.terminal_width)
        print(f"{'ID':<10} {'TESTADA':<10} {'CONECTADA':<12} {'PROCESSANDO':<12}")
        print("-" * self.terminal_width)
        
        for camera_id, status in self.system_monitor.camera_status.items():
            tested = "SIM" if status["connection_tested"] else "NÃO"
            connected = "SIM" if status["connection_established"] else "NÃO"
            processing = "SIM" if status["model_processing"] else "NÃO"
            
            print(f"{camera_id:<10} {tested:<10} {connected:<12} {processing:<12}")
            
        print()
        print("=" * self.terminal_width)
        print(f"Atualização automática a cada {self.update_interval} segundos")
        print(f"Pressione Ctrl+C para encerrar")
        print("=" * self.terminal_width)


# Código para executar o display de status em um processo separado
if __name__ == "__main__" and sys.argv[0].endswith("monitoring_display.py"):
    # Este código será executado quando o módulo for chamado como script para exibir status
    # Seria necessário passar os dados de status através de um arquivo ou comunicação entre processos
    print("Módulo de exibição de status iniciado")
    print("Este módulo deve ser chamado pelo sistema principal")
    time.sleep(5)
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

class LoggerManager:
    """
    Classe responsável por gerenciar a configuração do sistema de logging.
    Configura apenas o root logger com filter, formatter e handlers.
    """
    
    def __init__(self, log_dir="logs"):
        """
        Inicializa o gerenciador de logging.
        
        Args:
            log_dir: Diretório onde os logs serão armazenados
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def setup_root_logger(self, console_level=logging.INFO):
        """
        Configura o root logger com handlers, formatter e filter.
        
        Args:
            console_level: O nível de log para o console.
        """
        # Definir nome do arquivo de log com timestamp
        log_filename = f"buffet_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = self.log_dir / log_filename
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Limpar handlers existentes (caso já existam)
        if root_logger.handlers:
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
        
        # Criar console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        
        # Criar file handler para logs completos
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Definir formatter comum para todos os handlers
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Criar filter
        class PackageFilter(logging.Filter):
            def filter(self, record):
                # Exemplo: filtrar mensagens de dependências externas muito verbosas
                return not record.name.startswith("urllib3")
        
        # Adicionar filter apenas ao root logger
        root_logger.addFilter(PackageFilter())
        
        # Adicionar handlers ao root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Log de inicialização movido para cá para garantir que seja registrado
        logging.getLogger(__name__).debug("Sistema de logging inicializado e configurado.") 
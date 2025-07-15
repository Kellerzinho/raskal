import time
import logging
import requests
from threading import Lock

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ExternalAPI')

class ExternalAPIClient:
    """Cliente para enviar dados para a API externa"""
    
    def __init__(self, api_url: str, auth_token: str = None):
        self.api_url = api_url
        self.headers = {}
        if auth_token:
            self.headers['Authorization'] = f'Bearer {auth_token}'
        self.headers['Content-Type'] = 'application/json'
        self.lock = Lock()
        self.last_sync_time = None
        self.sync_interval = 5  # segundos
        
    def send_data(self, data: dict, endpoint: str = "") -> bool:
        """Envia dados para a API externa"""
        current_time = time.time()
        
        with self.lock:
            # Evita envio excessivo de dados
            if (self.last_sync_time is not None and 
                current_time - self.last_sync_time < self.sync_interval):
                return False
                
            try:
                response = requests.put(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=5
                )
                
                if response.status_code in (200, 201):
                    self.last_sync_time = current_time
                    return True
                else:
                    logger.error(f"Erro na resposta da API: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Erro na conexão com a API: {str(e)}")
                return False
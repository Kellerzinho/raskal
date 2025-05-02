# Sistema de Monitoramento de Buffet

Um sistema de visão computacional para monitoramento automático de níveis de ocupação de bandejas em buffets, utilizando câmeras ESP32Cam e modelos YOLOv11x pré-treinados.

## Visão Geral

O Sistema de Monitoramento de Buffet é uma solução completa para estabelecimentos gastronômicos que precisam monitorar o nível de ocupação de bandejas em tempo real. O sistema utiliza câmeras ESP32Cam estrategicamente posicionadas sobre as mesas de buffet, detecta automaticamente as bandejas e calcula a porcentagem de ocupação, gerando alertas quando é necessário repor os alimentos.

### Principais Recursos

- 🎯 **Detecção Automática**: Identifica bandejas de comida utilizando modelos YOLOv11x treinados especificamente para este fim
- 📊 **Monitoramento em Tempo Real**: Calcula a porcentagem de ocupação de cada bandeja em tempo real
- 🔔 **Alertas de Reposição**: Gera alertas quando o nível de ocupação está abaixo do limite configurado
- 📱 **Dashboard**: Integração com dashboard externo para visualização e gerenciamento
- 🔄 **API REST**: Permite integração com outros sistemas
- 📹 **Suporte Multi-câmera**: Gerencia várias câmeras simultaneamente
- 🖥️ **Visualização**: Interface para visualização dos frames processados

## Arquitetura do Sistema

O sistema é composto pelos seguintes módulos:

- **Main**: Coordena o fluxo de processamento e ciclo de vida da aplicação (`main.py`)
- **Camera Manager**: Gerencia conexões com câmeras ESP32Cam (`camera_manager.py`)
- **Vision**: Implementa o modelo YOLOv11x para detecção de objetos (`vision.py`)
- **Processing**: Processa as detecções e calcula métricas (`processing.py`)
- **API Server**: Fornece uma API para interação com o sistema (`api_server.py`)
- **API Thread**: Sincroniza dados com dashboard externo (`api_thread.py`)
- **External API**: Cliente para envio de dados (`externalAPI.py`)

## Requisitos de Sistema

- Python 3.8 ou superior
- PyTorch 2.0 ou superior
- CUDA (opcional, mas recomendado para melhor desempenho)
- Câmeras ESP32Cam configuradas com firmware adequado
- Pelo menos 4GB de RAM (8GB recomendado)
- Espaço em disco: mínimo de 2GB para o sistema e modelos

## Instalação

### Passo 1: Configurar ambiente Conda

Recomendamos o uso do Conda para gerenciar o ambiente Python. Caso não tenha o Conda instalado, baixe e instale o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/products/distribution).

```bash
# Criar um novo ambiente Conda
conda create -n buffet_monitor python=3.8
conda activate buffet_monitor
```

### Passo 2: Preparar diretório do projeto

```bash
# Crie um diretório para o projeto
mkdir buffet-monitoring-system
cd buffet-monitoring-system
```

### Passo 3: Instalar dependências

```bash
# Instalar PyTorch com suporte a CUDA (se disponível)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Instalar outras dependências
pip install -r requirements.txt
```

Caso não tenha uma GPU compatível com CUDA, instale a versão para CPU:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### Passo 4: Baixar o modelo pré-treinado

```bash
# Criar diretório para os modelos
mkdir -p models
cd models

# Baixar o modelo FVBM.pt (substitua pelo link real quando disponível)
wget https://example.com/models/FVBM.pt

cd ..
```

### Passo 5: Configurar as câmeras

Edite o arquivo `config/cameras.json` para adicionar suas câmeras ESP32Cam:

```json
{
  "cameras": [
    {
      "id": "cam1",
      "ip": "192.168.1.101",
      "port": 81,
      "location": "Buffet Principal",
      "max_fps": 15,
      "restaurant": "Restaurante ABC"
    },
    {
      "id": "cam2",
      "ip": "192.168.1.102", 
      "port": 81,
      "location": "Buffet de Entradas",
      "max_fps": 15,
      "restaurant": "Restaurante ABC"
    }
  ]
}
```

## Uso

### Iniciar o sistema

```bash
# Ativar o ambiente conda
conda activate buffet_monitor

# Iniciar com visualização
python main.py

# Iniciar sem interface de visualização
python main.py --no-display

# Especificar diretório de configuração alternativo
python main.py --config-path /caminho/para/config
```

### Verificar o status do sistema

O sistema exibe estatísticas no console em intervalos regulares, incluindo:
- Número de threads ativas
- Número de câmeras conectadas
- Número de bandejas monitoradas
- Alertas de reposição

### Integração com Dashboard

O sistema inicia automaticamente um servidor API na porta 3320 que permite:
- Obter dados atualizados através da rota `/api/data`
- Verificar o status do sistema através da rota `/api/status`

### Encerrar o sistema

O sistema pode ser encerrado com Ctrl+C no terminal ou através da rota API para obtenção de dados.

## Configuração Avançada

### Configuração do modelo YOLOv11x

É possível ajustar os parâmetros do modelo YOLOv11x editando os valores em `main.py`:

```python
self.vision_processor = YOLOProcessor(
    model_path="models/FVBM.pt",
    use_cuda=self.cuda_available,
    conf_threshold=0.5,  # Ajuste conforme necessário
    iou_threshold=0.45   # Ajuste conforme necessário
)
```

### Configuração da API externa

Para integração com dashboard externo, edite os parâmetros em `main.py`:

```python
self.external_api_config = {
    'EXTERNAL_API_URL': "http://seu-dashboard.com/api/replacements/updateReplacements",
    'SYNC_INTERVAL': 10,  # Intervalo em segundos
    'AUTH_TOKEN': "seu-token-de-autenticacao"  # Se necessário
}
```

## Estrutura do Projeto

```
buffet-monitoring-system/
├── api_server.py        # Servidor API para interação com o sistema
├── api_thread.py        # Thread para sincronização com dashboard externo
├── camera_manager.py    # Gerenciamento de conexões com câmeras
├── config/              # Diretório de configurações
│   └── cameras.json     # Configuração das câmeras
├── externalAPI.py       # Cliente para API externa
├── logs/                # Diretório para arquivos de log
├── main.py              # Módulo principal do sistema
├── models/              # Diretório para modelos pré-treinados
│   └── FVBM.pt          # Modelo YOLOv11x treinado para detecção de bandejas
├── processing.py        # Processamento de detecções e cálculo de métricas
└── vision.py            # Implementação do modelo YOLOv11x
```

## Solução de Problemas

### Câmeras não conectam

- Verifique se as câmeras estão ligadas e na mesma rede
- Confirme se os IPs e portas estão corretos em `config/cameras.json`
- Teste a conexão manualmente acessando `http://IP:PORTA/stream` no navegador

### Erros relacionados ao CUDA

- Verifique se sua GPU é compatível com CUDA
- Confirme se os drivers da NVIDIA estão atualizados
- Reinstale o PyTorch com a versão correta do CUDA para sua GPU

### Sistema lento ou com alto uso de CPU

- Reduza o parâmetro `max_fps` nas configurações das câmeras
- Desative câmeras não essenciais
- Considere usar uma máquina com GPU para melhor desempenho
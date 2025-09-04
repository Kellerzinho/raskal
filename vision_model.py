#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Visão Computacional - Sistema de Monitoramento de Buffet
Responsável por implementar e configurar o modelo YOLOv11x pré-treinado.
"""

import logging
from pathlib import Path
import torch


class YOLOProcessor:
    """
    Classe responsável por carregar e gerenciar o modelo YOLOv11x para detecção de objetos.
    """
    
    def __init__(self, model_path="models/FVBM.pt", use_cuda=False, conf_threshold=0.5, iou_threshold=0.45, retina_masks=True):
        """
        Inicializa o processador YOLO.
        
        Args:
            model_path: Caminho para o arquivo do modelo pré-treinado
            use_cuda: Se True, utiliza GPU para processamento
            conf_threshold: Limiar de confiança para detecções
            iou_threshold: Limiar de IOU para supressão de não-máximos
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.use_cuda = use_cuda
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = None
        self.model = None
        self.class_names = []
        self.retina_masks = retina_masks
        
        # Inicializar o modelo
        self._load_model()
        
    def _load_model(self):
        """
        Carrega o modelo YOLOv11x pré-treinado e configura o dispositivo.
        """
        self.logger.info(f"Carregando modelo YOLOv11x de {self.model_path}")
        
        try:
            # Verificar se o arquivo do modelo existe
            if not self.model_path.exists():
                self.logger.error(f"Arquivo do modelo não encontrado: {self.model_path}")
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            # Configurar dispositivo
            if self.use_cuda and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.logger.info(f"Utilizando GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Utilizando CPU para inferência")
            
            # Carregar modelo
            try:
                # Tenta carregar usando Ultralytics YOLO
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    self.logger.info("Modelo carregado via Ultralytics YOLO")
                except ImportError:
                    # Fallback para PyTorch nativo
                    self.model = torch.load(self.model_path, map_location=self.device)
                    self.logger.info("Modelo carregado via PyTorch nativo")
                
                self.logger.info("Modelo YOLOv11x carregado com sucesso")
                
            except Exception as e:
                self.logger.error(f"Erro ao carregar modelo: {e}")
                raise
                
        except Exception as e:
            self.logger.exception(f"Falha ao inicializar o modelo YOLOv11x: {e}")
            raise

    def process_frame(self, frame):
        """
        Processa um único frame com o modelo YOLO.
        
        Args:
            frame: O frame da imagem a ser processado.
            
        Returns:
            Os resultados da detecção do modelo.
        """
        if self.model:
            # Ativa máscaras retina para segmentação em resolução do frame
            try:
                return self.model(frame, verbose=False, retina_masks=self.retina_masks)
            except TypeError:
                # Compatibilidade caso a versão da lib não aceite o parâmetro
                return self.model(frame, verbose=False)
        self.logger.warning("Modelo de visão não está carregado. Retornando None.")
        return None

    def get_annotated_frame(self, results):
        """
        Método legado não utilizado: a anotação final passou para o DetectionProcessor.
        Mantido por compatibilidade de interface, retorna None.
        """
        self.logger.debug("get_annotated_frame não é mais utilizado; anotações são feitas no DetectionProcessor")
        return None
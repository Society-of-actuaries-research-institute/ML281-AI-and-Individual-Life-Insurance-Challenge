# -*- coding: utf-8 -*-
"""
Knowledge Base Loader

Loads all knowledge base files at startup.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from .config import (
    KNOWLEDGE_BASE_DIR, EDA_DIR, CALIBRATION_DIR, SEGMENTS_DIR, SHAP_DIR
)


class KnowledgeBase:
    """
    Centralized knowledge base loader.
    
    Loads all JSON/text files from knowledge_base/ directory.
    """
    
    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = Path(kb_path) if kb_path else KNOWLEDGE_BASE_DIR
        self._loaded = False
        
        # Placeholders
        self.numerical_summary: Optional[Dict] = None
        self.categorical_summary: Optional[Dict] = None
        self.overall_ae: Optional[Dict] = None
        self.yearly_ae: Optional[Dict] = None
        self.global_shap: Optional[Dict] = None
        self.coverage_registry: Optional[Dict] = None
        self.spotlight_summary: Optional[Dict] = None
        self.spotlight_details: Dict[int, Dict] = {}
    
    def load_all(self) -> 'KnowledgeBase':
        """Load all knowledge base files."""
        if self._loaded:
            return self
        
        print(f"Loading knowledge base from: {self.kb_path}")
        
        # EDA
        self.numerical_summary = self._load_json('eda/numerical_summary.json')
        self.categorical_summary = self._load_json('eda/categorical_summary.json')
        
        # Calibration
        self.overall_ae = self._load_json('calibration/overall_ae.json')
        self.yearly_ae = self._load_json('calibration/yearly_ae.json')
        
        # SHAP
        self.global_shap = self._load_json('shap/global_importance.json')
        
        # Segments
        self.coverage_registry = self._load_json('segments/coverage_registry.json')
        self.spotlight_summary = self._load_json('segments/spotlight_summary.json')
        self._load_spotlight_details()
        
        self._loaded = True
        self._print_status()
        
        return self
    
    def _load_json(self, relative_path: str) -> Optional[Dict]:
        """Load a JSON file."""
        try:
            path = self.kb_path / relative_path
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON error in {relative_path}: {e}")
            return None
    
    def _load_spotlight_details(self):
        """Load all spotlight detail files."""
        spotlight_dir = self.kb_path / 'segments' / 'spotlight'
        if not spotlight_dir.exists():
            return
        
        for f in spotlight_dir.glob('SPOT_*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if 'segment_id' in data:
                        self.spotlight_details[data['segment_id']] = data
                        print(f"  [OK] Loaded Spotlight: {data['segment_id']}")
            except (json.JSONDecodeError, KeyError):
                continue
    
    def _print_status(self):
        """Print loading status."""
        items = [
            ('EDA Numerical', self.numerical_summary),
            ('EDA Categorical', self.categorical_summary),
            ('Overall A/E', self.overall_ae),
            ('Yearly A/E', self.yearly_ae),
            ('Global SHAP', self.global_shap),
            ('Coverage Registry', self.coverage_registry),
            ('Spotlight Summary', self.spotlight_summary),
        ]
        
        for name, obj in items:
            status = '[OK]' if obj else '[--]'
            print(f"  {status} {name}")

        print(f"  [OK] Spotlight Details: {len(self.spotlight_details)} files")
    
    def get_overall_rate(self) -> float:
        """Get overall death rate from calibration."""
        if self.overall_ae:
            # Try different possible keys
            for key in ['overall_death_rate', 'overall_rate', 'death_rate']:
                if key in self.overall_ae:
                    return self.overall_ae[key]
            # Calculate from total if available
            if 'total_actual_deaths' in self.overall_ae and 'total_expected_deaths' in self.overall_ae:
                # Approximate from A/E
                pass
        return 0.0097  # Default fallback


# Singleton instance
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base(kb_path: Optional[str] = None) -> KnowledgeBase:
    """Get or create the knowledge base singleton."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase(kb_path).load_all()
    return _kb_instance

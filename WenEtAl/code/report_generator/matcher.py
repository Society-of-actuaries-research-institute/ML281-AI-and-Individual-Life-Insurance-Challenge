# -*- coding: utf-8 -*-
"""
Segment Matcher

Matches input records to Coverage and Spotlight segments.
"""

from typing import Dict, Any, Optional, Tuple
from .knowledge import KnowledgeBase


class SegmentMatcher:
    """
    Matches input records to pre-defined segments.
    
    Uses decision tree leaf IDs to match:
    - Coverage segments (Layer A): Baseline risk segments
    - Spotlight segments (Layer B): Anomalous segments
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
    
    def match_coverage(self, leaf_id: int) -> Optional[Dict]:
        """
        Get coverage segment by leaf ID.
        
        Args:
            leaf_id: Leaf ID from coverage_tree.apply()
            
        Returns:
            Coverage segment dict or None
        """
        if not self.kb.coverage_registry:
            return None
        
        for seg in self.kb.coverage_registry.get('segments', []):
            if seg.get('leaf_id') == leaf_id:
                return seg
        return None
    
    def match_spotlight(self, leaf_id: int) -> Optional[Dict]:
        """
        Get spotlight segment by leaf ID (if anomalous).
        
        Args:
            leaf_id: Leaf ID from spotlight_tree.apply()
            
        Returns:
            Spotlight segment dict or None (if not anomalous)
        """
        return self.kb.spotlight_details.get(leaf_id)
    
    def match_both(
        self, 
        coverage_leaf: int, 
        spotlight_leaf: int
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Match both coverage and spotlight segments.
        
        Returns:
            (coverage_segment, spotlight_segment)
        """
        return (
            self.match_coverage(coverage_leaf),
            self.match_spotlight(spotlight_leaf)
        )
    
    def get_segment_context(
        self,
        coverage_segment: Optional[Dict],
        spotlight_segment: Optional[Dict]
    ) -> str:
        """
        Generate segment context string for prompt.
        
        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        
        if coverage_segment:
            lines.append("## Coverage Segment")
            lines.append(f"- Segment ID: {coverage_segment.get('segment_id')}")
            lines.append(f"- Rule: {coverage_segment.get('rule_text')}")
            stats = coverage_segment.get('statistics', {})
            lines.append(f"- A/E Ratio: {stats.get('ae_ratio')}")
            interp = coverage_segment.get('interpretation', {})
            if interp.get('risk'):
                lines.append(f"- Risk interpretation: {interp['risk']}")
            if interp.get('model'):
                lines.append(f"- Model calibration: {interp['model']}")

        if spotlight_segment:
            lines.append("\n## Spotlight Segment (Exception Alert)")
            lines.append(f"- Segment ID: {spotlight_segment.get('segment_id')}")
            lines.append(f"- Risk Level: {spotlight_segment.get('risk_level')}")
            lines.append(f"- Rule: {spotlight_segment.get('rule_text')}")
            stats = spotlight_segment.get('statistics', {})
            lines.append(f"- A/E Ratio: {stats.get('ae_ratio')}")
            interp = spotlight_segment.get('interpretation', {})
            if interp.get('risk'):
                lines.append(f"- Risk interpretation: {interp['risk']}")

        return "\n".join(lines) if lines else "No matching segment found"


def create_matcher(kb: KnowledgeBase) -> SegmentMatcher:
    """Factory function for SegmentMatcher."""
    return SegmentMatcher(kb)

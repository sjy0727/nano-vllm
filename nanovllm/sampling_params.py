from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    # 基础采样参数
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    
    # 生成控制
    max_tokens: int = 64
    min_tokens: Optional[int] = None
    ignore_eos: bool = False
    
    # 重复惩罚
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    
    # 停止条件
    stop: Optional[list[str]] = None
    stop_token_ids: Optional[list[int]] = None
    
    # 其他参数
    seed: Optional[int] = None
    use_beam_search: bool = False
    best_of: Optional[int] = None
    n: int = 1  # 生成数量
    
    def __post_init__(self):
        """验证参数值的有效性"""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.min_tokens is not None and self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be between -2.0 and 2.0")
        
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be between -2.0 and 2.0")
        
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        
        if self.n <= 0:
            raise ValueError("n must be positive")
        
        if self.best_of is not None and self.best_of < self.n:
            raise ValueError("best_of must be >= n")

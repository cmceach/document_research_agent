import os
from typing import Dict, List, Any, Tuple, Optional, Literal, Union
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.callbacks import get_openai_callback

# Import utility functions
from src.llm_calls.utils import format_context, format_previous_queries, clean_query_results, truncate_context_for_tokens

# ... existing code ... 
from .client.chat_completions import ChatCompletions
from .client.classify import Classify
from .client.embeddings import Embeddings
from .client.images import Images
from .client.speech import Speech
from .client.transcriptions import Transcriptions
from .client.translations import Translations

__all__ = ["ChatCompletions", "Classify", "Embeddings", "Images", "Speech", "Transcriptions", "Translations"]

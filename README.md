# NimbRo API

[![CI](https://github.com/ptzld/nimbro-api/actions/workflows/ci.yml/badge.svg)](https://github.com/ptzld/nimbro-api/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nimbro-api.svg)](https://pypi.org/project/nimbro-api)
[![Supported Versions](https://img.shields.io/pypi/pyversions/nimbro-api.svg)](https://pypi.org/project/nimbro-api)
[![Downloads](https://static.pepy.tech/badge/nimbro-api/month)](https://pepy.tech/project/nimbro-api)
[![Docs](https://img.shields.io/badge/docs-latest-success.svg)](https://github.com/ptzld/nimbro-api#readme)

**NimbRo API** is a robust and flexible framework for API clients in Python.

---

## ✨ Features

#### 🚢 Ships with clients for the following APIs:

##### OpenAI:
- [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) ([nimbro_api.openai.ChatCompletions](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/openai/client/chat_completions.py))
- [Embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create) ([nimbro_api.openai.Embeddings](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/openai/client/embeddings.py))
- [Images](https://developers.openai.com/api/reference/resources/images/methods/generate) ([nimbro_api.openai.Images](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/openai/client/images.py))
- [Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech) ([nimbro_api.openai.Speech](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/openai/client/speech.py))
- [Transcriptions](https://platform.openai.com/docs/api-reference/audio/createTranscription) ([nimbro_api.openai.Transcriptions](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/openai/client/transcriptions.py))
- [Translations](https://platform.openai.com/docs/api-reference/audio/createTranslation) ([nimbro_api.openai.Translations](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/openai/client/translations.py))

##### NimbRo Vision Servers:
- [Describe Anything Model](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/dam) ([nimbro_api.nimbro_vision_servers.Dam](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/nimbro_vision_servers/client/dam.py))
- [Florence-2](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/florence2) ([nimbro_api.nimbro_vision_servers.Florence2](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/nimbro_vision_servers/client/florence2.py))
- [Kosmos-2](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/kosmos2) ([nimbro_api.nimbro_vision_servers.Kosmos2](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/nimbro_vision_servers/client/kosmos2.py))
- [MM-Grounding-DINO and LLMDet](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/mmgroundingdino) ([nimbro_api.nimbro_vision_servers.MmGroundingDino](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/nimbro_vision_servers/client/mmgroundingdino.py))
- [Segment Anything Model 2 real-time](https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/sam2_realtime) ([nimbro_api.nimbro_vision_servers.Sam2Realtime](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/nimbro_vision_servers/client/sam2_realtime.py))

##### Miscellaneous:
- [No-as-a-Service](https://github.com/hotheadhacker/no-as-a-service) ([nimbro_api.misc.No](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/misc/client/no.py))
- [VLM-GIST](https://vlm-gist.github.io) ([nimbro_api.misc.VlmGist](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/api/misc/client/vlm_gist.py))

#### 🪧 All clients feature:
- Standardized interfaces
- Sensible default settings
- Monitoring of operation success
- Configurable retry behavior and timeouts
- Extensive documentation, logging, and tests

#### 🪧 In addition, some clients feature:
- Response caching
- Response healing
- Batching of large requests
- API generalization across providers

#### ⚙️ Global [settings and utilities](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/core/core.py) are exposed at the package level.<br>

#### 🛠️ Custom clients can be implemented using the [nimbro_api.Client](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/client/client.py) and [nimbro_api.ClientBase](https://github.com/ptzld/nimbro-api/blob/main/src/nimbro_api/client/client_base.py) classes.

---

### 🚀 Quick Start

Install this package:
```console
pip install nimbro-api
```

Set the API key for the provider you want to use (e.g. `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `VLLM_API_KEY`):
```console
export OPENROUTER_API_KEY='supersecretkey'
```

Create and use a [ChatCompletions](https://platform.openai.com/docs/api-reference/chat) client:
```python
import nimbro_api
# nimbro_api.set_api_key(name="OPENROUTER_API_KEY", key="supersecretkey") # alternatively, set API key here
client = nimbro_api.openai.ChatCompletions(endpoint="OpenRouter", model="google/gemini-3-flash-preview")
success, message, response = client.prompt(text="Tell me a joke!")
```

---

## 📖 Citation

If you utilize this package in your research, please cite one of our relevant publications.

* **Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking**<br>
    [[arXiv:2503.16538](https://arxiv.org/abs/2503.16538)]
    ```bibtex
    @article{paetzold25vlmgist,
        author={Bastian P{\"a}tzold and Jan Nogga and Sven Behnke},
        title={Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking},
        journal={IEEE Robotics and Automation Letters (RA-L)},
        volume={10},
        number={11},
        pages={11578-11585},
        year={2025}
    }
    ```

* **A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics**<br>
    [[arXiv:2410.22997](https://arxiv.org/abs/2410.22997)]
    ```bibtex
    @article{bode24prompt,
        author={Jonas Bode and Bastian P{\"a}tzold and Raphael Memmesheimer and Sven Behnke},
        title={A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics},
        journal={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
        pages={309-314},
        year={2024}
    }
    ```

* **RoboCup@Home 2024 OPL Winner NimbRo: Anthropomorphic Service Robots using Foundation Models for Perception and Planning**<br>
    [[arXiv:2412.14989](https://arxiv.org/abs/2412.14989)]
    ```bibtex
    @article{memmesheimer25robocup,
        author={Raphael Memmesheimer and Jan Nogga and Bastian P{\"a}tzold and Evgenii Kruzhkov and Simon Bultmann and Michael Schreiber and Jonas Bode and Bertan Karacora and Juhui Park and Alena Savinykh and Sven Behnke},
        title={{RoboCup@Home 2024 OPL Winner NimbRo}: Anthropomorphic Service Robots using Foundation Models for Perception and Planning},
        journal={RoboCup 2024: RoboCup World Cup XXVII},
        volume={15570},
        pages={515-527},
        year={2025}
    }
    ```

---

## 📄 License

**NimbRo API** is licensed under the BSD-3-Clause License.

---

## 👤 Author

Bastian Pätzold – <paetzoldbastian@gmail.com>

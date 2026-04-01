from nimbro_api.client import Client
from ..base.vlm_gist_base import VlmGistBase

default_settings = {
    'logger_severity': None,
    'logger_name': "VLM-GIST",
    'message_results': True,
    'include_image': False,
    'scene_description': {
        'skip': True,
        'chat_completions': {
            'logger_severity': "warn"
        },
        'system_prompt_role': "system",
        'system_prompt': "You are a robot's visual perception system that identifies and analyzes objects in an image. Be concise and factual.",
        'image_prompt_role': "user",
        'image_prompt_detail': "high",
        'description_prompt_role': "user",
        'description_prompt':
            "Please describe the content of the image above. "
            "Focus your your description on all visible objects. "
            "Be concise and answer with at most 10 sentences. "
            "If you are unsure about identifying an object, make one single guess rather than calling it an unknown object or discussing eventualities."
    },
    'structured_description': {
        'skip': False,
        'chat_completions': {
            'logger_severity': "warn"
        },
        'use_scene_description': False,
        'system_prompt_role': "system",
        'system_prompt': "You are a robot's visual perception system that identifies and analyzes objects in an image. Be concise and factual.",
        'image_prompt_role': "user",
        'image_prompt_detail': "high",
        'description_prompt_role': "user",
        'description_prompt':
            "Provide a list in JSON format that contains each object (including furniture, persons, and animals) visible in the image above. "
            "Explicitly include each object instance as an individual list element, and never group multiple instances that are clearly distinct from one another. "
            "Each list element must be a dictionary with the fields object_name and description. "
            "The object_name of all humans must be person."
            "The description must be a single short sentence (max. 10 words, starting with 'A' or 'An'), "
            "that differs from the other descriptions and summarizes the most important information about the type, color, and appearance of the object, "
            "allowing for a visual identification of the object without knowing any of the descriptions generated for the other objects.",
        'response_type': "json",
        'keys_required': ['object_name', 'description'],
        'keys_required_types': ['str', 'str'],
        'keys_optional': [],
        'keys_optional_types': []
    },
    'detection': {
        'skip': False,
        'extract_from_description': False,
        'mmgroundingdino': {
            'logger_severity': "warn",
            'message_results': False,
            'nms_iou': None
        },
        'prompt_key': "description"
    },
    'segmentation': {
        'skip': False,
        'track': False,
        'sam2_realtime': {
            'logger_severity': "warn"
        }
    },
    'batch_size': 0,
    'batch_style': "multiprocessing",
    'batch_logger_severity': "warn",
    'retry': 1
}

class VlmGist(Client):
    """
    This is an implementation of VLM-GIST (https://arxiv.org/abs/2503.16538), with sensible default settings and behaviors throughout,
    extensive capabilities for configuring endpoints, models, and stages, managing connections, converting image encodings, and logging.

    "Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking" by Pätzold, Nogga & Behnke. Robotics and Automation Letters. 2025.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing VLM-GIST (https://arxiv.org/abs/2503.16538).
        """
        super().__init__(client_base=VlmGistBase, settings=settings, default_settings=default_settings, **kwargs)

    def get_settings(self, name=None):
        """
        Obtain all settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings:
            logger_severity (str | None):
                Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] (str) or `None` to adopt global process-wide severity.
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            message_results (bool):
                Include results in successful response messages when using `get_descriptions()`.
            include_image (bool):
                Include the Base64-encoded image data in the result dictionary returned by `run()`.
            scene_description (dict):
                Settings for the scene description step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - chat_completions (dict): Settings forwarded to `ChatCompletions`. See its `get_settings()`.
                - system_prompt_role (str): Role of the system prompt message. One of ["system", "user"].
                - system_prompt (str): Content of the system prompt message.
                - image_prompt_role (str): Role of the image prompt message. One of ["system", "user"].
                - image_prompt_detail (str): Detail level of the image prompt. One of ["high", "low", "auto"].
                - description_prompt_role (str): Role of the description prompt message. One of ["system", "user"].
                - description_prompt (str): Content of the description prompt message.
            structured_description (dict):
                Settings for the structured description step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - chat_completions (dict): Settings forwarded to `ChatCompletions`. See its `get_settings()`.
                - use_scene_description (bool): Prepend the scene description to the structured description prompt as context.
                - system_prompt_role (str): Role of the system prompt message. One of ["system", "user"].
                - system_prompt (str): Content of the system prompt message.
                - image_prompt_role (str): Role of the image prompt message. One of ["system", "user"].
                - image_prompt_detail (str): Detail level of the image prompt. One of ["high", "low", "auto"].
                - description_prompt_role (str): Role of the description prompt message. One of ["system", "user"].
                - description_prompt (str): Content of the description prompt message.
                - response_type (str): Expected response format. One of ["json", "text"].
                - keys_required (list[str]): Non-empty list of required keys expected in each object of the structured description.
                - keys_required_types (list[str]): Types for each required key, parallel to 'keys_required'. Each element must be one of ["str", "bool"].
                - keys_optional (list[str]): List of optional keys that may appear in each object of the structured description.
                - keys_optional_types (list[str]): Types for each optional key, parallel to 'keys_optional'. Each element must be one of ["str", "bool"].
            detection (dict):
                Settings for the object detection step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - mmgroundingdino (dict): Settings forwarded to `MmGroundingDino`. See its `get_settings()`.
                - prompt_key (str): Key from 'keys_required' in 'structured_description' whose value is used as the detection prompt for each object.
            segmentation (dict):
                Settings for the instance segmentation step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - track (bool): After initializing SAM2 with detections, run a second inference pass on the same image to obtain tracked masks. Must be `False` when 'skip' is `True`.
                - sam2_realtime (dict): Settings forwarded to `Sam2Realtime`. See its `get_settings()`.
            batch_size (int):
                Number of parallel workers when processing a list of images (>= 0). Use `0` to spawn one worker per image.
            batch_style (str):
                Parallelization backend used for batch processing. One of ["threading", "multiprocessing"].
            batch_logger_severity (str | None):
                Logger severity applied to each worker in ["debug", "info", "warn", "error", "fatal", "off"] (str) or `None` to adopt global process-wide severity.
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry:
                - If `True`, retries indefinitely. If `False`, failure is returned immediately.
                - Use a positive integer (`int`) to permit a specific number of retry attempts.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            any: A deep copy of the current settings (`dict`) or a single setting when providing 'name' (`any`).

        Notes:
            - See the global dictionary 'default_settings' on top of this file for defaults.
        """
        return self._base.get_settings(name)

    def set_settings(self, settings=None, **kwargs):
        """
        Configure all settings or a subset of them.

        Args:
            settings (dict | None, optional):
                New settings to apply. Settings not contained are kept.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Use `None` to reset all settings to their initial values. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
        """
        return self._base.wrap(0, self._base.set_settings, settings, **kwargs)

    def run(self, image, *, scene_description=None, structured_description=None, detection=None, **kwargs):
        """
        Run VLM-GIST on an image.

        Args:
            image (str | bytes | dict | list[str] | list[bytes], list[dict]):
                The image file to be processed as a local path, URL, Base64 encoding (all `str`), raw `bytes`, or `list` thereof.
                Images (`str` or `bytes`) can be embedded in a dictionary (`dict` or `list[dict]`), using the key 'data', allowing all other items to be included in the result.
            scene_description (str | dict | None, optional):
                If provided, continues a partial result from a previous call to `run()`.
                If provided, requires generation of this step to be skipped (setting 'scene_description.skip'). Defaults to `None`.
            structured_description (list[dict] | dict | None, optional):
                If provided, continues a partial result from a previous call to `run()`.
                If provided, reequires generation of this step to be skipped (setting 'structured_description.skip'). Defaults to `None`.
            detection (list[dict] | dict | None, optional):
                If provided, continues a partial result from a previous call to `run()`.
                If provided, requires generation of this step to be skipped (setting 'detection.skip'). Defaults to `None`.

            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[str] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - dict[str] | None: A dictionary (`dict`) containing the results, or `None` if not successful.

        Notes:
            - Structure of resulting dictionary:
                - run: stamp (ISO 8601 at start): str, type (normal/batch/worker): str, settings: dict, success (depending on type): bool, message: str, duration (seconds): float
                - image: stamp (ISO 8601 at start): str, success: bool, logs: list[str], data (base64): str, path: str, duration (seconds): float
                - scene_description: stamp (ISO 8601 at start): str, hash: str, settings: dict, success: bool, logs: list[str], usage: dict, raw (deleted when matching data): str, data: str, duration (seconds): float
                - structured_description: stamp (ISO 8601 at start): str, hash: str, settings: dict, logs: list[str], success: bool, usage: dict, raw (deleted when matching data): list[dict], data: list[dict], duration (seconds): float
                - detection: stamp (ISO 8601 at start): str, settings: dict, hash: str, success: bool, logs: list[str], raw (deleted when matching data): list[dict], data: list[dict], duration (seconds): float
                - segmentation: stamp (ISO 8601 at start): str, settings: dict, hash: str, success: bool, logs: list[str], raw (deleted when matching data): list[dict], data: list[dict], duration_init (initialization in seconds when also tracking): float, duration (seconds): float
                - batch: list[dict]
            - Key 'prompt' in 'detection' corresponds to the key selected via 'prompt_key' in 'structured_description'.
            - Key 'track_id' in 'segmentation' correspond to the 'detection' index.
        """
        is_worker = kwargs.pop('is_worker', False)
        return self._base.wrap(1, self._base.run, image, scene_description, structured_description, detection, is_worker, **kwargs)

from nimbro_api.client import Client
from ..base.vlm_gist_base import VlmGistBase

default_settings = {
    'logger_severity': None,
    'logger_name': "VLM-GIST",
    'message_process': True,
    'message_results': False,
    'include_image': False,
    'scene_description': {
        'skip': True,
        'message_process': True,
        'message_results': False,
        'chat_completions': {
            'logger_severity': "warn",
            'endpoint': "OpenRouter",
            'model': "google/gemini-3.1-flash-lite",
            'choices': 1,
            'reasoning_effort': "none",
        },
        'system_prompt_role': "system",
        'system_prompt': "You are a visual perception system that identifies and analyzes objects and other visible features in an image.",
        'image_prompt_role': "user",
        'image_prompt_detail': "high",
        'description_prompt_role': "user",
        'description_prompt':
            "Please provide a complete and detailed description of the image above. Your description must explicitly address all visible objects or features, "
            "even if they are small or in the background. Do not leave out or ignore anything. Your answer should consist of at least fifteen and no more than twenty-five sentences. "
            "If you are unsure about identifying an object, make a single educated guess instead of describing it as an unknown object or discussing different possibilities."
    },
    'structured_description': {
        'skip': False,
        'message_process': True,
        'message_results': False,
        'chat_completions': {
            'logger_severity': "warn",
            'endpoint': "OpenRouter",
            'model': "google/gemini-3.1-flash-lite",
            'choices': 1,
            'reasoning_effort': "none"
        },
        'use_scene_description': False,
        'system_prompt_role': "system",
        'system_prompt': "You are a visual perception system that identifies and analyzes objects and other visible features in an image.",
        'image_prompt_role': "user",
        'image_prompt_detail': "high",
        'description_prompt_role': "user",
        'description_prompt':
            "Provide a list in JSON format that contains each object (including furniture, persons, animals, etc.) visible in the image above. "
            "Explicitly include each object instance without exception as an individual list element. "
            "Never group multiple instances that are clearly distinct from one another. "
            "Each list element must be a dictionary with the fields label, description, and box_2d. "
            "The key label must specify the general object category or class (one or two words), so that similar objects across multiple images can be grouped by it."
            "The key description must contain an analysis of the object comprising at least ten and no more than twenty sentences that describe its properties (type, brand, model, state, function, etc.), appearance (material, design, color, texture, wear, labels, writing, etc.), and relation to other objects (position, orientation, attachment, proximity, etc.) in great detail. This description must be self-contained and not refer to any of the other descriptions, so that finding the corresponding object in the image is possible purely from reading it. "
            "The key box_2d must contain a bounding box of the object in [y_min, x_min, y_max, x_max] format normalized to pixel coordinates from 0 to 1000. "
            "Make sure to not ignore any object, do not forget any of the required keys, and follow the per-key instructions exactly. "
            "Be sure to include not only the objects in the center of the image, but absolutely everything, regardless of its size or location. "
            "If you're unsure about an object's nature, include it anyway to the best of your ability.",
        'response_type': "json",
        'keys_required': ['label', 'description', 'box_2d'],
        'keys_required_types': ['str', 'str', 'box_yxyx[int1000]'],
        'keys_optional': [],
        'keys_optional_types': []
    },
    'detection': {
        'skip': False,
        'message_process': True,
        'message_results': False,
        'extract_from_description': True,
        'prompt_key': "label",
        'mmgroundingdino': {
            'logger_severity': "warn",
            'message_results': False,
            'nms_iou': None,
            'overdetect_factor': 0.0
        },
        'allow_incomplete': False,
        'allow_excessive': False
    },
    'segmentation': {
        'skip': True,
        'message_process': True,
        'message_results': False,
        'track': False,
        'sam2_realtime': {
            'logger_severity': "warn"
        },
        'allow_incomplete': False,
        'allow_excessive': False
    },
    'batch': {
        'logger_severity': "info",
        'size': 8,
        'style': "threading",
        'retry': 2
    },
    'retry': 2
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

        Args:
            settings (dict | None, optional):
                Settings initializing the object. Settings not contained are initialized to their default values.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Nested settings can be specified using dot-separated keys (e.g., "a.b.c" is equivalent to {"a": {"b": {"c": ...}}}).
                Use `None` to initialize with default settings. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be initialized via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.
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
                Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] (`str`) or `None` to adopt global process-wide severity.
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            message_process (bool):
                Emit an info log when starting to process an image.
            message_results (bool):
                Include results in successful response messages when using `run()`.
            include_image (bool):
                Include the Base64-encoded image data in the result dictionary returned by `run()`.
            scene_description (dict):
                Settings for the scene description step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - message_process (bool): Emit an info log before and after a scene description step.
                - message_results (bool): Include results in the logs emitted after a scene description step.
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
                - message_process (bool): Emit an info log before and after a structured description step.
                - message_results (bool): Include results in the logs emitted after a structured description step.
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
                - keys_required_types (list[str]): Types for each required key, parallel to 'keys_required'.
                  Each element must be one of ["str", "bool", "int", "likert5", "likert7", "float", "unit", "list", "point_xy[int]", "point_yx[int]", "point_xy[int1000]", "point_yx[int1000]", "box_xyxy[int]", "box_yxyx[int]", "box_xyxy[int1000]", "box_yxyx[int1000]"].
                  Note that all 'int1000' types are unnormalized to the absolute image dimensions automatically, which is reflected in the settings returned with the result, stating the regular 'int' version of the type instead of the set 'int1000' type.
                - keys_optional (list[str]): List of optional keys that may appear in each object of the structured description.
                - keys_optional_types (list[str]): Types for each optional key, parallel to 'keys_optional'.
                  Each element must be one of ["str", "bool", "int", "likert5", "likert7", "float", "unit", "list", "point_xy[int]", "point_yx[int]", "point_xy[int1000]", "point_yx[int1000]", "box_xyxy[int]", "box_yxyx[int]", "box_xyxy[int1000]", "box_yxyx[int1000]"].
                  Note that all 'int1000' types are unnormalized to the absolute image dimensions automatically, which is reflected in the settings returned with the result, stating the regular 'int' version of the type instead of the set 'int1000' type.
            detection (dict):
                Settings for the object detection step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - message_process (bool): Emit an info log before and after a detection step.
                - message_results (bool): Include results in the logs emitted after a detection step.
                - extract_from_description (bool): Extract bounding boxes from structured description instead of using the detector.
                  This required setting 'keys_required_types' to contain exactly one box type, while 'keys_optional_types' must not contain any box types.
                - prompt_key (str): Key from 'keys_required' in 'structured_description' whose value is used as the detection prompt for each object.
                - mmgroundingdino (dict): Settings forwarded to `MmGroundingDino`. See its `get_settings()`.
                - allow_incomplete (bool): If `False`, returns an error or triggers a retry unless every item in the structured description is detected.
                - allow_excessive (bool): If `False`, returns an error or triggers a retry when an item in the structured description is detected more than once.
            segmentation (dict):
                Settings for the instance segmentation step.
                - skip (bool): Skip this step. At least one step must not be skipped.
                - message_process (bool): Emit an info log before and after a segmentation step.
                - message_re sults (bool): Include results in the logs emitted after a segmentation step.
                - track (bool): After initializing SAM2 with detections, run a second inference pass on the same image to obtain tracked masks. Must be `False` when 'skip' is `True`.
                - sam2_realtime (dict): Settings forwarded to `Sam2Realtime`. See its `get_settings()`.
                - allow_incomplete (bool): If `False`, returns an error or triggers a retry unless every detection is segmented.
                - allow_excessive (bool): If `False`, returns an error or triggers a retry when a detection is segmented more than once.
            batch (dict):
                Settings for processing multiple images passed to `run()`. Ignored when not passing a list of images to `run()`.
                - logger_severity (str | None): Logger severity applied to each worker in ["debug", "info", "warn", "error", "fatal", "off"] (`str`) or `None` to adopt global process-wide severity.
                - size (int): Number of parallel workers when processing a list of images (>= 0). Use `0` to spawn one worker per image.
                - style (str): Parallelization backend used for batch processing. One of ["threading", "multiprocessing"].
                - retry (bool | int): Retry behavior for a single worker when passing multiple images.
                  Then, the global 'retry' setting is applied to the entire batch, triggering retry when at least one worker failed (after using up all their retry attempts).
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry:
                - If `True`, retries indefinitely. If `False`, failure is returned immediately.
                - Use a positive integer (`int`) to permit a specific number of retry attempts.
                - Scope depends on mode (see 'batch.retry').

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
                Nested settings can be specified using dot-separated keys (e.g., "a.b.c" is equivalent to {"a": {"b": {"c": ...}}}).
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

    def visualize(self, result, *, image=None, output_dir=None, vis_args=None, **kwargs):
        """
        Visualize the detection and segmentation results produced by `run()` on the corresponding image(s).

        Args:
            result (str | dict):
                A result returned by `run()` (for either a single image or a batch), either as the dictionary (`dict`) itself or as a path (`str`) to a JSON file containing it.
            image (str | bytes | list[str] | list[bytes] | None):
                The image(s) corresponding to the result as a local path to a file or a folder containing files, URL, Base64 encoding (all `str`), or raw `bytes`.
                For a batch result, a `list` of length matching the batch size is required. If `None`, the image must be provided
                within 'result', which requires the setting 'include_image' to have been enabled during the originating call to `run()`. Defaults to `None`.
            output_dir (str | None):
                If provided, the directory in which to save the rendered visualization(s) as PNG file(s).
                The directory is created if it does not exist. Filenames are prefixed with the item index and a timestamp.
                If `None`, visualizations are only returned and not written to disk. Defaults to `None`.
            vis_args (dict | None):
                If provided, a dictionary with keyword arguments forwarded to `nimbro_api.utility.visual.visualize_detections`. Defaults to `None`.

        Raises:
            UnrecoverableError: If the visual dependencies (`cv2`, `numpy`) are not available.

        Returns:
            tuple[bool, str, list[numpy.ndarray] | None, list[str] | None]: A tuple containing:
                - bool: `True` if all items were visualized successfully, `False` otherwise.
                - str: A descriptive message about the operation result.
                - list[numpy.ndarray] | None: A `list` of rendered visualizations (one per result item, `None` for items that failed), or `None` if none succeeded.
                - list[str] | None: A `list` of output file paths (one per result item, `None` for items that were not saved), or `None` if nothing was saved to disk.

        Notes:
            - Results without a successful detection step are discarded.
            - When a valid segmentation is present, both boxes and masks are drawn.
            - Labels are taken from the 'prompt' key of each detection item.
            - The first point attribute of each item in the structured description is visualized as well.
        """
        return self._base.wrap(2, self._base.visualize, result, image, output_dir, vis_args, **kwargs)

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

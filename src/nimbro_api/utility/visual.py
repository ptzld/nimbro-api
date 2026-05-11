import os
from importlib.resources import files

import nimbro_api
from .io import decode_b64, parse_image_b64
from .misc import assert_type_value, assert_log

try:
    import cv2
    import numpy as np
    from PIL import Image as ImagePIL
    from PIL import ImageDraw, ImageFont
except ImportError as e:
    IMPORT_ERROR = repr(e)
else:
    IMPORT_ERROR = None

class Color:
    """
    Represents a single named color with multiple color formats.

    Attributes:
        name (str): The name of the color (e.g. "red").
        hex (str): The hexadecimal color code (e.g. "#FF0000").
        rgb (Tuple[int, int, int]): The 8bit RGB representation as a tuple of integers.
        bgr (Tuple[int, int, int]): The 8bit BGR representation as a tuple of integers.
    """

    def __init__(self, hex_code, name="Color"):
        """
        Initializes a Color instance.

        Args:
            hex_code (str): The hex code of the color (e.g. "#AABBCC").
            name (str): The name of the color.

        Raises:
            UnrecoverableError: If input arguments are invalid.
        """
        # parse arguments
        assert_type_value(obj=hex_code, type_or_value=str, name="argument 'hex_code'")
        assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
        assert_log(expression=name not in ["name", "hex", "rgb", "bgr"], message=f"Color name refers to reserved keyword '{name}'.")

        self.name = name
        self.hex = hex_code
        self.rgb = self._hex_to_rgb(hex_code)
        self.bgr = self.rgb[::-1]

    def _hex_to_rgb(self, hex_code):
        hex_code = hex_code.lstrip("#")
        return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))

    def __repr__(self):
        return f"{self.name}(hex='{self.hex}', rgb={self.rgb}, bgr={self.bgr})"

class ColorPalette:
    """
    A collection of named colors with optional named subgroups.
    Colors and subgroups can be accessed as attributes or by key.

    Attributes:
        name (str): Name of the palette.
        names (Tuple[str]): All color names in the palette.
        hex (Tuple[str]): All hex codes in the palette.
        hex_shuffle (Tuple[str]): All hex codes in the palette in random order.
        rgb (Tuple[Tuple[int, int, int]]): All 8bit RGB tuples in the palette.
        rgb_shuffle (Tuple[Tuple[int, int, int]]): All 8bit RGB tuples in the palette in random order.
        bgr (Tuple[Tuple[int, int, int]]): All 8bit BGR tuples in the palette.
        bgr_shuffle (Tuple[Tuple[int, int, int]]): All 8bit BGR tuples in the palette in random order.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Notes:
        - Example:
            ```python
            palette = ColorPalette({
                "red": "#FF0000",
                "green": "#00FF00",
                "blue": "#0000FF"
            }, groups={"primary": ["red", "blue"]})
            palette.red.rgb     # (255, 0, 0)
            palette.primary.hex # ("#FF0000", "#0000FF")
            ```
    """
    def __init__(self, colors, name="ColorPalette", groups=None):
        """
        Initialize a ColorPalette.

        Args:
            colors (dict[str, str]): Mapping of color names to hex codes.
            name (str, optional): Name of the ColorPalette.
            groups (dict[str, list[str]] | None, optional): Mapping of group names to lists of color names.

        Raises:
            UnrecoverableError: If input arguments are invalid.
        """
        # parse arguments
        assert_type_value(obj=colors, type_or_value=dict, name="argument 'colors'")
        assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
        assert_type_value(obj=groups, type_or_value=[dict, None], name="argument 'groups'")

        self.name = name
        self.names = []
        self._colors = []
        assert_log(expression=len(colors) > 0, message="Palette must at least contain one color.")
        for name, hex_code in colors.items():
            assert_log(expression=name not in ["name", "names", "hex", "rgb", "bgr", "groups"], message=f"Color name refers to reserved keyword '{name}'.")
            self.names.append(name)
            self._colors.append(Color(hex_code=hex_code, name=name))
            setattr(self, name, self._colors[-1])
        self.names = tuple(self.names)
        self._colors = tuple(self._colors)

        self.groups = {}
        if groups:
            for group_name, group_keys in groups.items():
                assert_log(expression=group_name not in ["name", "names", "hex", "rgb", "bgr"], message=f"Group name refers to reserved keyword '{group_name}'.")
                assert_log(expression=group_name not in colors, message=f"Group name refers to known color '{group_name}'.")
                for key in group_keys:
                    assert_log(expression=key in colors, message=f"Group name '{group_name}' refers to unknown color '{key}'.")
                group_dict = {k: colors[k] for k in group_keys}
                subgroup = ColorPalette(colors=group_dict, name=f"{self.name}.{group_name}")
                setattr(self, group_name, subgroup)
                self.groups[group_name] = tuple(group_keys)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._colors[key]
        elif isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
        raise KeyError(f"Invalid key: {key}")

    @property
    def hex(self):  # noqa: A003
        return tuple(c.hex for c in self._colors)

    @property
    def hex_shuffle(self):
        colors = [c.hex for c in self._colors]
        np.random.shuffle(colors)
        return tuple(colors)

    @property
    def rgb(self):
        return tuple(c.rgb for c in self._colors)

    @property
    def rgb_shuffle(self):
        colors = [c.rgb for c in self._colors]
        np.random.shuffle(colors)
        return tuple(colors)

    @property
    def bgr(self):
        return tuple(c.bgr for c in self._colors)

    @property
    def bgr_shuffle(self):
        colors = [c.bgr for c in self._colors]
        np.random.shuffle(colors)
        return tuple(colors)

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        if len(self.groups) > 0:
            colors = "\n\t\t" + ",\n\t\t".join(repr(c) for c in self._colors) + "\n\t"
            groups = "\n\t\t" + ",\n\t\t".join(f"{name}{repr(self.groups[name]).replace("'", "")}" for name in self.groups) + "\n\t" # noqa: E999
            return f"{self.name}(\n\tcolors: [{colors}],\n\tgroups: [{groups}]\n)"
        else:
            colors = "\n\t" + ",\n\t".join(repr(c) for c in self._colors) + "\n"
            return f"{self.name}([{colors}])"

nimbro_colors = ColorPalette(
    colors={
        'petrol': "#0C5678",
        'sun': "#FFB500",
        'violet': "#412163",
        'lime': "#9AB800",
        'red': "#900000",
        'sky': "#92B7CF",
        'teal': "#145050",
        'rose': "#FFD7B6",
        'salmon': "#E37092",
        'khaki': "#8C8240",
        'purple': "#B799FF",
        'surf': "#A3D590",
        'yellow': "#FFEA00",
        'blue': "#3975AC",
        'rosa': "#FFCADE",
        'cyan': "#009996",
        'blood': "#4F000A",
        'olive': "#455200",
        'pink': "#DA0050",
        'indigo': "#000D80",
        'orange': "#FF7300",
        'green': "#457400",
        'lila': "#8200BE",
        'mint': "#D2F5CC",
        'brown': "#663C00"
    },
    name="nimbro",
    groups={'ten': ['petrol', 'sun', 'violet', 'lime', 'red', 'sky', 'teal', 'rose', 'salmon', 'khaki']}
)

kelly_colors = ColorPalette(
    colors={
        'white': "#FdFDFD",
        'black': "#1D1D1D",
        'yellow': "#EBCE2B",
        'purple': "#702C8C",
        'orange': "#DB6917",
        'aqua': "#96CDE6",
        'red': "#BA1C30",
        'buff': "#C0BD7F",
        'gray': "#7F7E80",
        'green': "#5FA641",
        'pink': "#D485B2",
        'blue': "#4277B6",
        'papaya': "#DF8461",
        'violet': "#463397",
        'manilla': "#E1A11A",
        'plum': "#91218C",
        'lemon': "#E8E948",
        'brown': "#7E1510",
        'lime': "#92AE31",
        'dirt': "#6F340D",
        'crimson': "#D32B1E",
        'olive': "#2B3514"
    },
    name="kelly",
    groups={'accent': ['yellow', 'purple', 'orange', 'aqua', 'red', 'buff', 'green', 'pink', 'blue', 'papaya', 'violet', 'manilla', 'plum', 'lemon', 'brown', 'lime', 'dirt', 'crimson', 'olive']}
)

def visualize_detections(image, *, boxes=None, masks=None, labels=None, points=None, box_format="xyxy_normalized", point_format="xy_normalized", is_rgb=False, **kwargs):
    """
    Draws bounding boxes, masks, labels, and points on an image with customizable options.

    Args:
        image (str | bytes | numpy.ndarray):
            The image to be processed as a local file path, URL, Base64 encoding (all `str`), raw `bytes`, or cv2 style `numpy.ndarray`.
        boxes (list | tuple | numpy.ndarray | None, optional):
            A list of bounding boxes. Each box must be a 4-element sequence according to 'box_format'.
            Individual entries may also be `None` to indicate that a detection has no box.
            Defaults to `None`.
        masks (list | None, optional):
            A list of boolean masks corresponding to detections.
            Each mask must be of type `list`, `tuple`, `np.ndarray`, or `None`.
            Use `None` or a list containing `None` to not show masks. Defaults to `None`.
        labels (list | None, optional):
            List of string labels for each detection. Use `None` to deactivate all or a single label.
            Providing a label for a detection requires that the detection also has a box. Defaults to `None`.
        points (list | tuple | numpy.ndarray | None, optional):
            A list of points (one per detection) pointing to the object. Each point must be a
            2-element sequence (x, y) according to 'point_format', or `None` to indicate that a
            detection has no point. Defaults to `None`.
        box_format (str, optional):
            Format of the input boxes. See `convert_boxes()`.
            One of ["xyxy_normalized", "xyxy_absolute", "xywh_normalized", "xywh_absolute"].
            Defaults to 'xyxy_normalized'.
        point_format (str, optional):
            Format of the input points. One of ["xy_normalized", "xy_absolute"].
            For 'xy_normalized', coordinates are floats in [0.0, 1.0].
            For 'xy_absolute', coordinates are non-negative integer pixel coordinates.
            Defaults to 'xy_normalized'.
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (`True`) or BGR (`False`) order.
            Defaults to `False`.

        **kwargs:
            colors (str | list | tuple | dict):
                Defines the main colors of the visualization.
                    - Use 'auto' to automatically set a color per detection.
                    - Use 'auto_class' to automatically select a color per unique detection label.
                    - Pass a list of 3-tuples (tuple | list) defining an 8-bit BGR color per detection.
                    - Pass a dict mapping detection labels (str | None) to colors as 3-tuples (tuple | list) of 8-bit BGR integers.
                Defaults to 'auto_class'.
            alpha (float):
                Global opacity of the visualization in (0.0, 1.0]. Defaults to 0.9.
            box_thickness (float):
                Thickness of the boxes relative to the smaller image dimension in (0.0, 1.0].
                Defaults to 0.004.
            fill_alpha (float):
                Opacity of the filled box area in [0.0, 1.0]. Defaults to 0.0.
            box_alpha (float):
                Opacity of the box borders in [0.0, 1.0]. Defaults to 1.0.
            label_font_size (float):
                Font size relative to the smaller image dimension in (0.0, 1.0].
                Defaults to 0.022.
            label_font_path (str):
                Path to .ttf or .otf font file used.
                Defaults to 'DejaVuSans.ttf'.
            label_padding (float):
                Padding around box labels relative to the smaller image dimension in [0.0, 1.0].
                Defaults to 0.006.
            label_text_color (str | list | tuple | None):
                Defines the colors of the label texts.
                    - Use 'auto' to automatically select black or white per box based on contrast towards box color.
                    - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all label texts.
                    - Use `None` to adopt the color of the detection.
                Defaults to 'auto'.
            label_background_color (str | list | tuple | None):
                Defines the colors of the label backgrounds.
                    - Use 'auto' to adopt the color of the detection.
                    - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all label backgrounds.
                    - Use `None` to disable background behind the label texts.
                Defaults to 'auto'.
            label_text_alpha (float):
                Opacity of the label texts in (0.0, 1.0]. Defaults to 1.0.
            label_background_alpha (float):
                Opacity of the label background in [0.0, 1.0]. Defaults to 1.0.
            mask_color (str | list | tuple | None):
                Defines the colors of the masks.
                    - Use 'auto' to adopt the color of the detection.
                    - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all masks.
                    - Use `None` to adopt the color of the detection.
                Defaults to 'auto'.
            mask_alpha (float):
                Opacity of the masks in [0.0, 1.0]. Defaults to 0.25.
            contour_thickness (float):
                Thickness of mask contours relative to the smaller image dimension in [0.0, 1.0].
                Defaults to 0.0005.
            contour_color (str | list | tuple | None):
                Defines the colors of the mask contours.
                    - Use 'auto' to adopt the color of the detection.
                    - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all mask contours.
                    - Use `None` to adopt the color of the detection.
                Defaults to 'auto'.
            contour_alpha (float):
                Opacity of the contours around masks in [0.0, 1.0]. Defaults to 0.6.
            point_radius (float):
                Radius of the point marker relative to the smaller image dimension in [0.0, 1.0].
                Defaults to 0.007.
            point_color (str | list | tuple | None):
                Defines the fill colors of the points.
                    - Use 'auto' to adopt the color of the detection.
                    - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all points.
                    - Use `None` to adopt the color of the detection.
                Defaults to 'auto'.
            point_alpha (float):
                Opacity of the point fill in [0.0, 1.0]. Defaults to 0.9.
            point_outline_thickness (float):
                Thickness of the point outline relative to the smaller image dimension in [0.0, 1.0].
                Use 0.0 to disable the outline. Defaults to 0.0005.
            point_outline_color (str | list | tuple | None):
                Defines the colors of the point outlines.
                    - Use 'auto' to automatically select black or white per point based on contrast towards the point fill color.
                    - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all point outlines.
                    - Use `None` to adopt the color of the detection.
                Defaults to 'auto'.
            point_outline_alpha (float):
                Opacity of the point outline in [0.0, 1.0]. Defaults to 0.9.
            auto_color_palette (ColorPalette):
                Color palette used for automatic color selection. Defaults to `nimbro_colors`.
            auto_color_shuffle (bool):
                Shuffle automatic color selection. Defaults to `False`.
            draw_order (str):
                Order in which detections are drawn in ['size', 'input'],
                where 'size' uses the bounding box area in descending order
                and 'input' iterates detections as is. Defaults to 'size'.
            mask_format (str):
                Format of the input masks in ['box_local', 'full_image'].
                Use 'box_local' for masks in the spatial extent of the corresponding box.
                Use 'full_image' for masks in the spatial extent of the full image.
                Defaults to 'box_local'.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        numpy.ndarray: A BGR uint8 image of the same resolution visualizing the detections.
    """
    # validate availability
    assert_log(expression=IMPORT_ERROR is None, message=f"Visual utilities are not available due to missing dependencies: {IMPORT_ERROR}")

    # parse arguments
    if not isinstance(image, np.ndarray):
        success, message, image, _ = parse_image_b64(image=image)
        assert_log(expression=success, message=message)

        success, message, image = decode_b64(string=image, name="image")
        assert_log(expression=success, message=message)

        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        is_rgb = False # cv2 explicitly decodes as BGR

    assert_type_value(obj=box_format, type_or_value=["xyxy_normalized", "xyxy_absolute", "xywh_normalized", "xywh_absolute"], name="argument 'box_format'")
    assert_type_value(obj=point_format, type_or_value=["xy_normalized", "xy_absolute"], name="argument 'point_format'")
    assert_type_value(obj=boxes, type_or_value=[list, tuple, np.ndarray, None], name="argument 'boxes'")
    assert_type_value(obj=masks, type_or_value=[list, tuple, None], name="argument 'masks'")
    assert_type_value(obj=labels, type_or_value=[list, tuple, None], name="argument 'labels'")
    assert_type_value(obj=points, type_or_value=[list, tuple, np.ndarray, None], name="argument 'points'")
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")

    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    if isinstance(points, np.ndarray):
        points = points.tolist()

    num_boxes = 0 if boxes is None else len(boxes)
    num_masks = 0 if masks is None else len(masks)
    num_labels = 0 if labels is None else len(labels)
    num_points = 0 if points is None else len(points)
    num_detections = max(num_boxes, num_masks, num_labels, num_points)

    if boxes is None:
        boxes = [None] * num_detections
    else:
        assert_log(expression=len(boxes) == num_detections, message=f"Expected number of values in argument 'boxes' to match the number of detections '{num_detections}' but got '{len(boxes)}'.")

    if masks is None:
        masks = [None] * num_detections
    else:
        assert_log(expression=len(masks) == num_detections, message=f"Expected number of values in argument 'masks' to match the number of detections '{num_detections}' but got '{len(masks)}'.")

    if labels is None:
        labels = [None] * num_detections
    else:
        assert_log(expression=len(labels) == num_detections, message=f"Expected number of values in argument 'labels' to match the number of detections '{num_detections}' but got '{len(labels)}'.")

    if points is None:
        points = [None] * num_detections
    else:
        assert_log(expression=len(points) == num_detections, message=f"Expected number of values in argument 'points' to match the number of detections '{num_detections}' but got '{len(points)}'.")

    if box_format in ["xyxy_normalized", "xywh_normalized"]:
        for i, box in enumerate(boxes):
            if box is None:
                continue
            assert_type_value(obj=box, type_or_value=[list, tuple], name=f"item '{i}' in argument 'boxes'")
            assert_log(expression=len(box) == 4, message=f"Expected item '{i}' in argument 'boxes' to contain '4' values but got '{len(box)}'.")
            for j, value in enumerate(box):
                assert_type_value(obj=value, type_or_value=float, name=f"item '{j}' in box '{i}' argument 'boxes'")
                assert_log(expression=value >= 0.0 and value <= 1.0, message=f"{value}")
    else:
        for i, box in enumerate(boxes):
            if box is None:
                continue
            assert_type_value(obj=box, type_or_value=[list, tuple], name=f"item '{i}' in argument 'boxes'")
            assert_log(expression=len(box) == 4, message=f"Expected item '{i}' in argument 'boxes' to contain '4' values but got '{len(box)}'.")
            for j, value in enumerate(box):
                assert_type_value(obj=value, type_or_value=int, name=f"item '{j}' in box '{i}' argument 'boxes'")
                assert_log(expression=value >= 0, message=f"{value}")
            assert_log(expression=box[0] >= 0, message=f"Expected value in argument 'boxes' for x0 '{box[0]}' >= 0.")
            assert_log(expression=box[1] >= 0, message=f"Expected value in argument 'boxes' for y0 '{box[1]}' >= 0.")
            assert_log(expression=box[2] > 0, message=f"Expected value in argument 'boxes' for x1/w '{box[2]}' > 0.")
            assert_log(expression=box[3] > 0, message=f"Expected value in argument 'boxes' for y1/h '{box[3]}' > 0.")
            if box_format == "xyxy_absolute":
                assert_log(expression=box[0] < box[2], message=f"Expected value in argument 'boxes' for x0 '{box[0]}' < x1 '{box[2]}'.")
                assert_log(expression=box[1] < box[3], message=f"Expected value in argument 'boxes' for y0 '{box[1]}' < y1 '{box[3]}'.")
                assert_log(expression=box[2] <= image.shape[1], message=f"Expected value in argument 'boxes' for x1 '{box[2]}' <= image width '{image.shape[1]}'.")
                assert_log(expression=box[3] <= image.shape[0], message=f"Expected value in argument 'boxes' for y1 '{box[3]}' <= image height '{image.shape[0]}'.")
            else:
                assert_log(expression=box[0] + box[2] <= image.shape[1], message=f"Expected value in argument 'boxes' for x0 '{box[0]}' + width '{box[2]}' <= image width '{image.shape[1]}'.")
                assert_log(expression=box[1] + box[3] <= image.shape[0], message=f"Expected value in argument 'boxes' for y0 '{box[1]}' + height '{box[3]}' <= image height '{image.shape[0]}'.")

    if point_format == "xy_normalized":
        for i, point in enumerate(points):
            if point is None:
                continue
            assert_type_value(obj=point, type_or_value=[list, tuple], name=f"item '{i}' in argument 'points'")
            assert_log(expression=len(point) == 2, message=f"Expected item '{i}' in argument 'points' to contain '2' values but got '{len(point)}'.")
            for j, value in enumerate(point):
                assert_type_value(obj=value, type_or_value=float, name=f"item '{j}' in point '{i}' argument 'points'")
                assert_log(expression=value >= 0.0 and value <= 1.0, message=f"Expected value in argument 'points' to be in [0.0, 1.0] but got '{value}'.")
    else:
        for i, point in enumerate(points):
            if point is None:
                continue
            assert_type_value(obj=point, type_or_value=[list, tuple], name=f"item '{i}' in argument 'points'")
            assert_log(expression=len(point) == 2, message=f"Expected item '{i}' in argument 'points' to contain '2' values but got '{len(point)}'.")
            for j, value in enumerate(point):
                assert_type_value(obj=value, type_or_value=int, name=f"item '{j}' in point '{i}' argument 'points'")
                assert_log(expression=value >= 0, message=f"Expected value in argument 'points' to be non-negative but got '{value}'.")
            assert_log(expression=point[0] < image.shape[1], message=f"Expected value in argument 'points' for x '{point[0]}' < image width '{image.shape[1]}'.")
            assert_log(expression=point[1] < image.shape[0], message=f"Expected value in argument 'points' for y '{point[1]}' < image height '{image.shape[0]}'.")

    for i, label in enumerate(labels):
        assert_type_value(obj=label, type_or_value=[str, None], name=f"item '{i}' in argument 'labels'")

    for i in range(num_detections):
        if labels[i] is not None:
            assert_log(expression=boxes[i] is not None, message=f"Expected a box for detection '{i}' because a label was provided.")

    colors = kwargs.pop('colors', "auto_class")
    alpha = kwargs.pop('alpha', 0.9)
    box_thickness = kwargs.pop('box_thickness', 0.004)
    fill_alpha = kwargs.pop('fill_alpha', 0.0)
    box_alpha = kwargs.pop('box_alpha', 1.0)
    label_font_size = kwargs.pop('label_font_size', 0.022)
    label_font_path = kwargs.pop("label_font_path", str(files("nimbro_api").joinpath("fonts", "DejaVuSans.ttf")))
    label_padding = kwargs.pop('label_padding', 0.006)
    label_text_color = kwargs.pop('label_text_color', "auto")
    label_background_color = kwargs.pop('label_background_color', "auto")
    label_text_alpha = kwargs.pop('label_text_alpha', 1.0)
    label_background_alpha = kwargs.pop('label_background_alpha', 1.0)
    mask_color = kwargs.pop('mask_color', "auto")
    mask_alpha = kwargs.pop('mask_alpha', 0.25)
    contour_thickness = kwargs.pop('contour_thickness', 0.0005)
    contour_color = kwargs.pop('contour_color', "auto")
    contour_alpha = kwargs.pop('contour_alpha', 0.6)
    point_radius = kwargs.pop('point_radius', 0.007)
    point_color = kwargs.pop('point_color', "auto")
    point_alpha = kwargs.pop('point_alpha', 0.9)
    point_outline_thickness = kwargs.pop('point_outline_thickness', 0.0005)
    point_outline_color = kwargs.pop('point_outline_color', "auto")
    point_outline_alpha = kwargs.pop('point_outline_alpha', 0.9)
    auto_color_palette = kwargs.pop('auto_color_palette', nimbro_colors if num_detections > 10 else nimbro_colors.ten)
    auto_color_shuffle = kwargs.pop('auto_color_shuffle', False)
    draw_order = kwargs.pop('draw_order', "size")
    mask_format = kwargs.pop('mask_format', "box_local")
    assert_log(expression=len(kwargs) == 0, message=f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'.")

    assert_type_value(obj=colors, type_or_value=["auto", "auto_class", list, tuple, dict, None], name="argument 'colors'")
    if isinstance(colors, (list, tuple)):
        assert_log(expression=len(colors) == num_detections, message=f"Expected number of values in arguments 'boxes' and 'colors' to match but got '{num_detections}' and '{len(colors)}'.")
        for i, color in enumerate(colors):
            assert_type_value(obj=color, type_or_value=[list, tuple], name=f"item '{i}' in argument 'colors'")
            assert_log(expression=len(color) == 3, message=f"Expected all colors in argument 'colors' to contain '3' values but got '{len(color)}'.")
            for j, value in enumerate(color):
                assert_type_value(obj=value, type_or_value=int, name=f"item '{j}' in color '{i}' in argument 'colors'")
                assert_log(expression=0 <= value <= 255, message=f"Expected item '{j}' in color '{i}' in argument 'colors' to be 8-bit values but got '{value}'.")
    elif isinstance(colors, dict):
        if None not in colors:
            colors[None] = (0, 0, 0)
        for key in colors:
            assert_type_value(obj=key, type_or_value=[str, None], name="all keys in argument 'colors'")
            assert_type_value(obj=colors[key], type_or_value=[list, tuple], name=f"value of key '{key}' in argument 'colors'")
            assert_log(expression=len(colors[key]) == 3, message=f"Expected all colors in argument 'colors' to contain '3' values but got '{len(colors[key])}'.")
            for i, value in enumerate(colors[key]):
                assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in value of key '{key}' in argument 'colors'")
                assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'colors' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=alpha, type_or_value=float, name="argument 'alpha'")
    assert_log(expression=alpha > 0, message=f"Expected value of argument 'alpha' to be greater zero but got '{alpha}'.")
    assert_log(expression=alpha <= 1, message=f"Expected value of argument 'alpha' to be one or less but got '{alpha}'.")
    assert_type_value(obj=box_thickness, type_or_value=float, name="argument 'box_thickness'")
    assert_log(expression=box_thickness > 0, message=f"Expected value of argument 'box_thickness' to be greater zero but got '{box_thickness}'.")
    assert_log(expression=box_thickness <= 1, message=f"Expected value of argument 'box_thickness' to be one or less but got '{box_thickness}'.")
    assert_type_value(obj=fill_alpha, type_or_value=float, name="argument 'fill_alpha'")
    assert_log(expression=fill_alpha >= 0, message=f"Expected value of argument 'fill_alpha' to be zero or greater but got '{fill_alpha}'.")
    assert_log(expression=fill_alpha <= 1, message=f"Expected value of argument 'fill_alpha' to be one or less but got '{fill_alpha}'.")
    assert_type_value(obj=box_alpha, type_or_value=float, name="argument 'box_alpha'")
    assert_log(expression=box_alpha >= 0, message=f"Expected value of argument 'box_alpha' to be zero or greater but got '{box_alpha}'.")
    assert_log(expression=box_alpha <= 1, message=f"Expected value of argument 'box_alpha' to be one or less but got '{box_alpha}'.")
    assert_type_value(obj=label_font_size, type_or_value=float, name="argument 'label_font_size'")
    assert_log(expression=label_font_size > 0, message=f"Expected value of argument 'label_font_size' to be greater zero but got '{label_font_size}'.")
    assert_log(expression=label_font_size <= 1, message=f"Expected value of argument 'label_font_size' to be one or less but got '{label_font_size}'.")
    assert_type_value(obj=label_font_path, type_or_value=str, name="argument 'label_font_path'")
    assert_type_value(obj=label_padding, type_or_value=float, name="argument 'label_padding'")
    assert_log(expression=label_padding >= 0, message=f"Expected value of argument 'label_padding' to be zero or greater but got '{label_padding}'.")
    assert_log(expression=label_padding <= 1, message=f"Expected value of argument 'label_padding' to be one or less but got '{label_padding}'.")
    assert_type_value(obj=label_text_color, type_or_value=["auto", list, tuple, None], name="argument 'label_text_color'")
    if isinstance(label_text_color, (list, tuple)):
        assert_type_value(obj=label_text_color, type_or_value=[list, tuple], name="argument 'label_text_color'")
        assert_log(expression=len(label_text_color) == 3, message=f"Expected argument 'label_text_color' to contain '3' values but got '{len(label_text_color)}'.")
        for i, value in enumerate(label_text_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'label_text_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'label_text_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=label_background_color, type_or_value=["auto", list, tuple, None], name="argument 'label_background_color'")
    if isinstance(label_background_color, (list, tuple)):
        assert_type_value(obj=label_background_color, type_or_value=[list, tuple], name="argument 'label_background_color'")
        assert_log(expression=len(label_background_color) == 3, message=f"Expected argument 'label_background_color' to contain '3' values but got '{len(label_background_color)}'.")
        for i, value in enumerate(label_background_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'label_background_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'label_background_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=label_text_alpha, type_or_value=float, name="argument 'label_text_alpha'")
    assert_log(expression=label_text_alpha >= 0, message=f"Expected value of argument 'label_text_alpha' to be zero or greater but got '{label_text_alpha}'.")
    assert_log(expression=label_text_alpha <= 1, message=f"Expected value of argument 'label_text_alpha' to be one or less but got '{label_text_alpha}'.")
    assert_type_value(obj=label_background_alpha, type_or_value=float, name="argument 'label_background_alpha'")
    assert_log(expression=label_background_alpha >= 0, message=f"Expected value of argument 'label_background_alpha' to be zero or greater but got '{label_background_alpha}'.")
    assert_log(expression=label_background_alpha <= 1, message=f"Expected value of argument 'label_background_alpha' to be one or less but got '{label_background_alpha}'.")
    assert_type_value(obj=mask_color, type_or_value=["auto", list, tuple, None], name="argument 'mask_color'")
    if isinstance(mask_color, (list, tuple)):
        assert_type_value(obj=mask_color, type_or_value=[list, tuple], name="argument 'mask_color'")
        assert_log(expression=len(mask_color) == 3, message=f"Expected argument 'mask_color' to contain '3' values but got '{len(mask_color)}'.")
        for i, value in enumerate(mask_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'mask_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'mask_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=mask_alpha, type_or_value=float, name="argument 'mask_alpha'")
    assert_log(expression=mask_alpha >= 0, message=f"Expected value of argument 'mask_alpha' to be zero or greater but got '{mask_alpha}'.")
    assert_log(expression=mask_alpha <= 1, message=f"Expected value of argument 'mask_alpha' to be one or less but got '{mask_alpha}'.")
    assert_type_value(obj=contour_thickness, type_or_value=float, name="argument 'contour_thickness'")
    assert_log(expression=contour_thickness >= 0, message=f"Expected value of argument 'contour_thickness' to be zero or greater but got '{contour_thickness}'.")
    assert_log(expression=contour_thickness <= 1, message=f"Expected value of argument 'contour_thickness' to be one or less but got '{contour_thickness}'.")
    assert_type_value(obj=contour_color, type_or_value=["auto", list, tuple, None], name="argument 'contour_color'")
    if isinstance(contour_color, (list, tuple)):
        assert_type_value(obj=contour_color, type_or_value=[list, tuple], name="argument 'contour_color'")
        assert_log(expression=len(contour_color) == 3, message=f"Expected argument 'contour_color' to contain '3' values but got '{len(contour_color)}'.")
        for i, value in enumerate(contour_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'contour_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'contour_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=contour_alpha, type_or_value=float, name="argument 'contour_alpha'")
    assert_log(expression=contour_alpha >= 0, message=f"Expected value of argument 'contour_alpha' to be zero or greater but got '{contour_alpha}'.")
    assert_log(expression=contour_alpha <= 1, message=f"Expected value of argument 'contour_alpha' to be one or less but got '{contour_alpha}'.")
    assert_type_value(obj=point_radius, type_or_value=float, name="argument 'point_radius'")
    assert_log(expression=point_radius > 0, message=f"Expected value of argument 'point_radius' to be greater zero but got '{point_radius}'.")
    assert_log(expression=point_radius <= 1, message=f"Expected value of argument 'point_radius' to be one or less but got '{point_radius}'.")
    assert_type_value(obj=point_color, type_or_value=["auto", list, tuple, None], name="argument 'point_color'")
    if isinstance(point_color, (list, tuple)):
        assert_log(expression=len(point_color) == 3, message=f"Expected argument 'point_color' to contain '3' values but got '{len(point_color)}'.")
        for i, value in enumerate(point_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'point_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'point_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=point_alpha, type_or_value=float, name="argument 'point_alpha'")
    assert_log(expression=point_alpha >= 0, message=f"Expected value of argument 'point_alpha' to be zero or greater but got '{point_alpha}'.")
    assert_log(expression=point_alpha <= 1, message=f"Expected value of argument 'point_alpha' to be one or less but got '{point_alpha}'.")
    assert_type_value(obj=point_outline_thickness, type_or_value=float, name="argument 'point_outline_thickness'")
    assert_log(expression=point_outline_thickness >= 0, message=f"Expected value of argument 'point_outline_thickness' to be zero or greater but got '{point_outline_thickness}'.")
    assert_log(expression=point_outline_thickness <= 1, message=f"Expected value of argument 'point_outline_thickness' to be one or less but got '{point_outline_thickness}'.")
    assert_type_value(obj=point_outline_color, type_or_value=["auto", list, tuple, None], name="argument 'point_outline_color'")
    if isinstance(point_outline_color, (list, tuple)):
        assert_log(expression=len(point_outline_color) == 3, message=f"Expected argument 'point_outline_color' to contain '3' values but got '{len(point_outline_color)}'.")
        for i, value in enumerate(point_outline_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'point_outline_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'point_outline_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=point_outline_alpha, type_or_value=float, name="argument 'point_outline_alpha'")
    assert_log(expression=point_outline_alpha >= 0, message=f"Expected value of argument 'point_outline_alpha' to be zero or greater but got '{point_outline_alpha}'.")
    assert_log(expression=point_outline_alpha <= 1, message=f"Expected value of argument 'point_outline_alpha' to be one or less but got '{point_outline_alpha}'.")
    assert_type_value(obj=auto_color_palette, type_or_value=ColorPalette, name="argument 'auto_color_palette'")
    assert_type_value(obj=auto_color_shuffle, type_or_value=bool, name="argument 'auto_color_shuffle'")
    assert_type_value(obj=draw_order, type_or_value=["size", "input"], name="argument 'draw_order'")
    assert_type_value(obj=mask_format, type_or_value=["box_local", "full_image"], name="argument 'mask_format'")

    # convert to BGR
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_rgb = False

    # parse relative units
    unit = min(image.shape[:2])
    box_thickness = max(1, int(round(box_thickness * unit)))
    label_font_size = max(1, int(round(label_font_size * unit)))
    label_padding = max(0, int(round(label_padding * unit)))
    contour_thickness = max(0, int(round(contour_thickness * unit)))
    point_radius = max(1, int(round(point_radius * unit)))
    point_outline_thickness = max(0, int(round(point_outline_thickness * unit)))

    # read image
    if num_detections == 0:
        return image
    overlay = image.copy()

    # determine colors
    if colors == "auto":
        colors = []
        if auto_color_shuffle is True:
            while len(colors) < num_detections:
                colors += auto_color_palette.bgr[:num_detections] if len(auto_color_palette.bgr_shuffle) == 0 else auto_color_palette.bgr_shuffle[:num_detections]
        else:
            while len(colors) < num_detections:
                colors += auto_color_palette.bgr[:num_detections]
    elif colors == "auto_class":
        unique_labels = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)

        if len(unique_labels) == 0:
            unique_labels = [None]

        colors = []
        target_len = len(unique_labels)
        if auto_color_shuffle is True:
            while len(colors) < target_len:
                colors += auto_color_palette.bgr[:target_len] if len(auto_color_palette.bgr_shuffle) == 0 else auto_color_palette.bgr_shuffle[:target_len]
        else:
            while len(colors) < target_len:
                colors += auto_color_palette.bgr[:target_len]
        colors = dict(zip(unique_labels, colors))

    if isinstance(colors, dict):
        colors_per_box = []
        for i in range(num_detections):
            key = labels[i]
            assert_log(expression=key in colors, message=f"Expected a color for label '{key}' in argument 'colors'.")
            colors_per_box.append(colors[key])
        colors = colors_per_box

    # convert boxes
    valid_boxes = []
    valid_indices = []
    for i, box in enumerate(boxes):
        if box is None:
            continue
        valid_boxes.append(box)
        valid_indices.append(i)

    boxes_xyxy = [None] * num_detections
    if len(valid_boxes) > 0:
        valid_boxes = convert_boxes(boxes=valid_boxes, source_format=box_format, target_format="xyxy_absolute", image_size=image.shape[:2])
        for i, box in zip(valid_indices, valid_boxes):
            boxes_xyxy[i] = box
    boxes = boxes_xyxy

    # convert points to absolute pixel coordinates
    points_abs = [None] * num_detections
    h_img, w_img = image.shape[:2]
    for i, point in enumerate(points):
        if point is None:
            continue
        if point_format == "xy_normalized":
            px = int(round(point[0] * (w_img - 1)))
            py = int(round(point[1] * (h_img - 1)))
        else:
            px = int(point[0])
            py = int(point[1])
        px = max(0, min(w_img - 1, px))
        py = max(0, min(h_img - 1, py))
        points_abs[i] = (px, py)
    points = points_abs

    for i in range(len(masks)):
        assert_type_value(obj=masks[i], type_or_value=[list, tuple, np.ndarray, None], name=f"item '{i}' in argument 'masks'")
        if isinstance(masks[i], (list, tuple)):
            masks[i] = np.asarray(masks[i])
        if masks[i] is not None:
            assert_log(expression=masks[i].dtype == np.bool_, message=f"{masks[i].dtype}")

            if mask_format == "box_local":
                assert_log(expression=boxes[i] is not None, message=f"Expected a box for detection '{i}' because a box-local mask was provided.")
                x0, y0, x1, y1 = boxes[i]
                expected_shape = (y1 - y0, x1 - x0)
                assert_log(expression=masks[i].shape == expected_shape, message=f"{masks[i].shape} {expected_shape}")

                full_mask = np.zeros(image.shape[:2], dtype=bool)
                full_mask[y0:y1, x0:x1] = masks[i]
                masks[i] = full_mask
            else:
                assert_log(expression=masks[i].shape == image.shape[:2], message=f"{masks[i].shape} {image.shape[:2]}")

    # draw boxes and labels
    if draw_order == "size":
        sizes = []
        for i in range(num_detections):
            if boxes[i] is not None:
                sizes.append((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
            elif masks[i] is not None:
                sizes.append(int(np.count_nonzero(masks[i])))
            else:
                sizes.append(0)
        order = np.argsort(-np.array(sizes))
    elif draw_order == "input":
        order = range(num_detections)

    for i in order:
        if masks[i] is not None:
            if mask_alpha > 0:
                overlay_roi = overlay[masks[i]]
                target_color = np.array(colors[i] if mask_color in ["auto", None] else mask_color, dtype=np.float32)
                overlay[masks[i]] = np.clip((1 - mask_alpha) * overlay_roi + mask_alpha * target_color, 0, 255).astype(np.uint8)

            if contour_thickness > 0:
                contours, _ = cv2.findContours(
                    image=(masks[i] * 255).astype(np.uint8),
                    mode=cv2.RETR_LIST, # RETR_LIST RETR_EXTERNAL
                    method=cv2.CHAIN_APPROX_NONE # CHAIN_APPROX_NONE CHAIN_APPROX_SIMPLE
                )

                h, w = masks[i].shape[:2]
                contour_mask = np.zeros((h, w), dtype=np.uint8)
                color = colors[i] if contour_color in ["auto", None] else contour_color

                for contour in contours:
                    points_c = contour[:, 0]
                    lines_to_draw = []
                    for j in range(len(points_c)):
                        pt1 = tuple(points_c[j])
                        pt2 = tuple(points_c[(j + 1) % len(points_c)])
                        if any((
                            pt1[0] <= 0, pt1[0] >= w - 1, pt1[1] <= 0, pt1[1] >= h - 1,
                            pt2[0] <= 0, pt2[0] >= w - 1, pt2[1] <= 0, pt2[1] >= h - 1
                        )):
                            continue
                        lines_to_draw.append((pt1, pt2))

                    for pt1, pt2 in lines_to_draw:
                        cv2.line(contour_mask, pt1, pt2, color=255, thickness=contour_thickness, lineType=cv2.LINE_AA)

                contour_mask = contour_mask.astype(bool)
                overlay_roi = overlay[contour_mask]
                target_color = np.array(color, dtype=np.float32)
                overlay[contour_mask] = np.clip((1 - contour_alpha) * overlay_roi + contour_alpha * target_color, 0, 255).astype(np.uint8)

        if boxes[i] is not None:
            if fill_alpha > 0:
                x0, y0, x1, y1 = boxes[i]
                overlay_roi = overlay[y0:y1, x0:x1]
                target_color = np.array(colors[i], dtype=np.float32)
                overlay[y0:y1, x0:x1] = np.clip((1 - fill_alpha) * overlay_roi + fill_alpha * target_color, 0, 255).astype(np.uint8)

            if box_alpha > 0:
                overlay = draw_rectangle(image=overlay, box=boxes[i], box_format="xyxy_absolute", color=colors[i], thickness=box_thickness, alpha=box_alpha, is_rgb=is_rgb)

        if points[i] is not None:
            # determine fill color
            if point_color in ["auto", None]:
                fill_color = colors[i]
            else:
                fill_color = point_color

            # determine outline color
            if point_outline_color is None:
                outline_color = colors[i]
            elif point_outline_color == "auto":
                # calculate luminance using ITU-R BT.709 coefficients
                b, g, r = fill_color
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
                if luminance > 0.5:
                    outline_color = (0, 0, 0)
                else:
                    outline_color = (255, 255, 255)
            else:
                outline_color = point_outline_color

            # build a per-point mask for the fill (filled disk)
            h, w = overlay.shape[:2]
            if point_alpha > 0:
                fill_canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(fill_canvas, points[i], radius=point_radius, color=255, thickness=-1, lineType=cv2.LINE_AA)
                fill_mask = fill_canvas.astype(bool)
                if np.any(fill_mask):
                    # use soft alpha weights from the anti-aliased canvas
                    weights = (fill_canvas.astype(np.float32) / 255.0) * point_alpha
                    weights = weights[fill_mask][:, None]
                    overlay_roi = overlay[fill_mask].astype(np.float32)
                    target_color = np.array(fill_color, dtype=np.float32)
                    overlay[fill_mask] = np.clip((1 - weights) * overlay_roi + weights * target_color, 0, 255).astype(np.uint8)

            # outline (annular ring) drawn on top
            if point_outline_thickness > 0 and point_outline_alpha > 0:
                outline_canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(outline_canvas, points[i], radius=point_radius, color=255, thickness=point_outline_thickness, lineType=cv2.LINE_AA)
                outline_mask = outline_canvas.astype(bool)
                if np.any(outline_mask):
                    weights = (outline_canvas.astype(np.float32) / 255.0) * point_outline_alpha
                    weights = weights[outline_mask][:, None]
                    overlay_roi = overlay[outline_mask].astype(np.float32)
                    target_color = np.array(outline_color, dtype=np.float32)
                    overlay[outline_mask] = np.clip((1 - weights) * overlay_roi + weights * target_color, 0, 255).astype(np.uint8)

        if labels[i] is not None:
            display_text = labels[i]

            if label_text_color is None:
                text_color = colors[i]
            elif label_text_color == "auto":
                # calculate luminance using ITU-R BT.709 coefficients
                b, g, r = label_background_color if isinstance(label_background_color, (list, tuple)) else colors[i]
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
                if luminance > 0.5:
                    text_color = (0, 0, 0)
                else:
                    text_color = (255, 255, 255)
            else:
                text_color = label_text_color

            if label_background_color is None:
                background_color = None
            elif label_background_color == "auto":
                background_color = colors[i]
            else:
                background_color = label_background_color

            overlay = draw_text(
                image=overlay,
                text=display_text,
                anchor=(boxes[i][0] - box_thickness, boxes[i][1] - 1),
                font_path=label_font_path,
                font_size=label_font_size,
                text_color=text_color,
                background_color=background_color,
                padding=label_padding,
                line_gap=label_padding,
                is_rgb=is_rgb,
                background_alpha=label_background_alpha,
                text_alpha=label_text_alpha,
            )

    # blend the overlay with the original image
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image

def draw_rectangle(image, box, *, box_format="xyxy_normalized", is_rgb=False, **kwargs):
    """
    Draw an unfilled rectangle on an image.

    Args:
        image (str | bytes | numpy.ndarray):
            The image to be processed as a local file path, URL, Base64 encoding (all `str`), raw `bytes`, or cv2 style `numpy.ndarray`.
        box (tuple):
            A bounding box as a 4-element sequence according to `box_format`. See `convert_boxes()`.
        box_format (str, optional):
            Format of the input boxes. See `convert_boxes()`.
            One of ["xyxy_normalized", "xyxy_absolute", "xywh_normalized", "xywh_absolute"].
            Defaults to 'xyxy_normalized'
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (`True`) or BGR (`False`) order.
            Defaults to `False`.

        **kwargs:
            color (tuple | list):
                Color of the rectangle as 3-tuple (tuple | list) 8-bit BGR.
                Defaults to (255, 255, 255).
            thickness (int):
                Extension of the rectangle in pixels (>0) outward from `box`.
                Defaults to 1.
            alpha (float):
                Opacity of the rectangle in [0.0, 1.0].
                Defaults to 1.0.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        numpy.ndarray: A BGR uint8 image of the same resolution with the rectangle drawn on it.
    """
    # validate availability
    assert_log(expression=IMPORT_ERROR is None, message=f"Visual utilities are not available due to missing dependencies: {IMPORT_ERROR}")

    # parse arguments
    if not isinstance(image, np.ndarray):
        success, message, image, _ = parse_image_b64(image=image)
        assert_log(expression=success, message=message)

        success, message, image = decode_b64(string=image, name="image")
        assert_log(expression=success, message=message)

        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        is_rgb = False # cv2 explicitly decodes as BGR

    assert_type_value(obj=box, type_or_value=[list, tuple, np.ndarray], name="argument 'box'")
    assert_log(expression=len(box) == 4, message=f"Expected argument 'box' to contain '4' items but got '{len(box)}'.")

    valid_formats = ["xyxy_absolute", "xyxy_normalized", "xywh_absolute", "xywh_normalized"]
    assert_type_value(obj=box_format, type_or_value=valid_formats, name="argument 'box_format'")
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")
    color = kwargs.pop('color', (255, 255, 255))
    thickness = kwargs.pop('thickness', 1)
    alpha = kwargs.pop('alpha', 1.0)
    assert_log(expression=len(kwargs) == 0, message=f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'.")
    assert_type_value(obj=color, type_or_value=[list, tuple], name="argument 'color'")
    assert_log(expression=len(color) == 3, message=f"Expected argument 'color' to contain '3' values but got '{len(color)}'.")
    for i, value in enumerate(color):
        assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'color'")
        assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=thickness, type_or_value=int, name="argument 'thickness'")
    assert_log(expression=thickness > 0, message=f"Expected value of argument 'thickness' to be greater zero but got '{thickness}'.")
    assert_type_value(obj=alpha, type_or_value=float, name="argument 'alpha'")
    assert_log(expression=alpha >= 0, message=f"Expected value of argument 'alpha' to be zero or greater but got '{alpha}'.")
    assert_log(expression=alpha <= 1, message=f"Expected value of argument 'alpha' to be one or less but got '{alpha}'.")

    # convert to BGR
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if alpha == 0:
        return image
    h, w = image.shape[:2]

    # convert box
    x0, y0, x1, y1 = convert_boxes(boxes=box, source_format=box_format, target_format="xyxy_absolute", image_size=image.shape[:2])

    # create safe mathematical color array for blending
    color_arr = np.array(color, dtype=np.float32)

    # draw lines outward up to thickness
    for t in range(thickness):
        # horizontal lines (top and bottom)
        yt_top = y0 - (t + 1)
        yt_bottom = y1 + t

        x_start_h = max(0, x0 - (t + 1))
        x_end_h = min(w, x1 + t + 1)

        if 0 <= yt_top < h:
            roi = image[yt_top, x_start_h:x_end_h]
            image[yt_top, x_start_h:x_end_h] = np.clip((1 - alpha) * roi + alpha * color_arr, 0, 255).astype(np.uint8)

        if 0 <= yt_bottom < h:
            roi = image[yt_bottom, x_start_h:x_end_h]
            image[yt_bottom, x_start_h:x_end_h] = np.clip((1 - alpha) * roi + alpha * color_arr, 0, 255).astype(np.uint8)

        # vertical lines (left and right)
        xt_left = x0 - (t + 1)
        xt_right = x1 + t

        y_start_v = max(0, y0 - t)
        y_end_v = min(h, y1 + t)

        if 0 <= xt_left < w:
            roi = image[y_start_v:y_end_v, xt_left]
            image[y_start_v:y_end_v, xt_left] = np.clip((1 - alpha) * roi + alpha * color_arr, 0, 255).astype(np.uint8)

        if 0 <= xt_right < w:
            roi = image[y_start_v:y_end_v, xt_right]
            image[y_start_v:y_end_v, xt_right] = np.clip((1 - alpha) * roi + alpha * color_arr, 0, 255).astype(np.uint8)

    return image

def draw_text(image, text, *, anchor=(0, 0), is_rgb=False, **kwargs):
    """
    Draw a text on an image.

    Args:
        image (str | bytes | numpy.ndarray):
            The image to be processed as a local file path, URL, Base64 encoding (all `str`), raw `bytes`, or cv2 style `numpy.ndarray`.
        text (str):
            Text to be drawn.
        anchor (tuple | list, optional):
            Tuple of valid (x, y) pixel-coordinates defining the lower left corner of the drawn text.
            If the text box exceeds the boundaries of the image, the anchor is automatically clamped to fit.
            Defaults to (0, 0).
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (`True`) or BGR (`False`) order.
            Defaults to `False`.

        **kwargs:
            font_path (str):
                Path to .ttf or .otf font file used to draw text.
                Defaults to 'DejaVuSans.ttf'.
            font_size (int):
                Font size (>0) of the text. Defaults to 22.
            text_color (tuple | list):
                Color of the text as 3-tuple (tuple | list) 8-bit BGR.
                Defaults to (255, 255, 255).
            background_color (tuple | list | None):
                Color of the background behind the the text as 3-tuple (tuple | list) 8-bit BGR.
                Use `None` to not draw the background. Defaults to `None`.
            padding (int):
                Space between text and background on all sides in pixels (>=0).
                Defaults to 4.
            line_gap (int):
                Space between lines when text is a multiline-string in pixels (>=0).
                Defaults to 4.
            text_alpha (float):
                Opacity of the text in [0.0, 1.0].
                Defaults to 1.0.
            background_alpha (float):
                Opacity of the background in [0.0, 1.0].
                Defaults to 1.0.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        numpy.ndarray: A BGR uint8 image of the same resolution with the text drawn on it.
    """
    # validate availability
    assert_log(expression=IMPORT_ERROR is None, message=f"Visual utilities are not available due to missing dependencies: {IMPORT_ERROR}")

    # parse arguments
    if not isinstance(image, np.ndarray):
        success, message, image, _ = parse_image_b64(image=image)
        assert_log(expression=success, message=message)

        success, message, image = decode_b64(string=image, name="image")
        assert_log(expression=success, message=message)

        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        is_rgb = False # cv2 explicitly decodes as BGR

    assert_type_value(obj=text, type_or_value=str, name="argument 'text'")
    assert_type_value(anchor, [tuple, list], name="argument 'anchor'")
    assert_log(expression=len(anchor) == 2, message=f"Expected argument 'anchor' to be a 2-tuple (x, y) but got a tuple of length '{len(anchor)}'.")
    assert_log(expression=isinstance(anchor[0], (int, float)) and isinstance(anchor[1], (int, float)), message="Expected argument 'anchor' to be a 2-tuple (x, y) of numbers.")
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")
    font_path = kwargs.pop("font_path", str(files("nimbro_api").joinpath("fonts", "DejaVuSans.ttf")))
    font_size = kwargs.pop('font_size', 22)
    text_color = kwargs.pop('text_color', (255, 255, 255))
    background_color = kwargs.pop('background_color', None)
    padding = kwargs.pop('padding', 4)
    line_gap = kwargs.pop('line_gap', 4)
    text_alpha = kwargs.pop('text_alpha', 1.0)
    background_alpha = kwargs.pop('background_alpha', 1.0)
    assert_log(expression=len(kwargs) == 0, message=f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'.")
    assert_type_value(obj=font_path, type_or_value=str, name="argument 'font_path'")
    assert_type_value(obj=font_size, type_or_value=int, name="argument 'font_size'")
    assert_log(expression=font_size > 0, message=f"Expected 'font_size' to be greater zero but got '{font_size}'.")
    assert_type_value(obj=text_color, type_or_value=[list, tuple], name="argument 'text_color'")
    assert_log(expression=len(text_color) == 3, message=f"Expected argument 'text_color' to contain '3' values but got '{len(text_color)}'.")
    for i, value in enumerate(text_color):
        assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'text_color'")
        assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'text_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=background_color, type_or_value=[list, tuple, None], name="argument 'background_color'")
    if background_color is not None:
        assert_log(expression=len(background_color) == 3, message=f"Expected argument 'background_color' to contain '3' values but got '{len(background_color)}'.")
        for i, value in enumerate(background_color):
            assert_type_value(obj=value, type_or_value=int, name=f"item '{i}' in argument 'background_color'")
            assert_log(expression=0 <= value <= 255, message=f"Expected item '{i}' in argument 'background_color' to be 8-bit values but got '{value}'.")
    assert_type_value(obj=padding, type_or_value=int, name="argument 'padding'")
    assert_log(expression=padding >= 0, message=f"Expected 'padding' to be zero or greater but got '{padding}'.")
    assert_type_value(obj=line_gap, type_or_value=int, name="argument 'line_gap'")
    assert_log(expression=line_gap >= 0, message=f"Expected 'line_gap' to be zero or greater but got '{line_gap}'.")
    assert_type_value(obj=text_alpha, type_or_value=float, name="argument 'text_alpha'")
    assert_log(expression=0.0 <= text_alpha <= 1.0, message=f"Expected 'text_alpha' to be in [0.0, 1.0] but got '{text_alpha}'.")
    assert_type_value(obj=background_alpha, type_or_value=float, name="argument 'background_alpha'")
    assert_log(expression=0.0 <= background_alpha <= 1.0, message=f"Expected 'background_alpha' to be in [0.0, 1.0] but got '{background_alpha}'.")

    # convert to RGB
    if not is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    text_color_rgb = tuple(text_color)[::-1]
    if background_color is not None:
        background_color_rgb = tuple(background_color)[::-1]
    else:
        background_color_rgb = None

    # read image and setup a transparent overlay for correct alpha compositing
    image_rgba = ImagePIL.fromarray(image).convert("RGBA")
    overlay = ImagePIL.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(font_path, font_size)
    lines = text.split('\n')

    # measure text size
    bboxes = [font.getbbox(line) for line in lines]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    max_width = max(bbox[2] - bbox[0] for bbox in bboxes)
    total_text_height = sum(heights)

    # define background box
    box_width = max_width + 2 * padding
    box_height = total_text_height + 2 * padding + (len(lines) - 1) * line_gap
    x_anchor, y_anchor = int(anchor[0]), int(anchor[1])

    if x_anchor < 0:
        x_anchor = 0
    elif x_anchor > image.shape[1] - 1 - box_width:
        x_anchor = image.shape[1] - 1 - box_width

    if y_anchor < box_height:
        y_anchor = box_height
    elif y_anchor > image.shape[0] - 1:
        y_anchor = image.shape[0] - 1

    x0 = x_anchor
    y0 = y_anchor - box_height
    x1 = x_anchor + box_width
    y1 = y_anchor

    # draw background
    if background_color_rgb is not None and background_alpha > 0.0:
        fill = background_color_rgb + (int(255 * background_alpha),)
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    # draw text
    y_cursor = y0 + padding
    for i, line in enumerate(lines):
        bbox = bboxes[i]
        line_height = bbox[3] - bbox[1]
        y_text = y_cursor - bbox[1]
        x_text = x_anchor + padding - bbox[0]
        fill = text_color_rgb + (int(255 * text_alpha),)
        draw.text((x_text, y_text), line, font=font, fill=fill)
        y_cursor += line_height + line_gap

    # composite over original image
    result = ImagePIL.alpha_composite(image_rgba, overlay).convert("RGB")
    final_arr = np.array(result)

    # guarantee BGR output format as strictly defined in Docstring
    final_arr = cv2.cvtColor(final_arr, cv2.COLOR_RGB2BGR)
    return final_arr

def convert_boxes(boxes, *, source_format="xyxy_absolute", target_format="xywh_absolute", image_size=None):
    """
    Convert bounding boxes between different formalisms.

    Args:
        boxes (list | tuple | numpy.ndarray):
            Bounding boxes to convert. Each box must be a 4-element sequence.
            The input may be a a single box [...], a list of boxes: [[...], [...]], or arbitrarily nested lists/tuples/arrays of boxes.
        source_format (str, optional):
            Format of the input boxes. One of:
            - 'xyxy_absolute':
                [x_min, y_min, x_max, y_max] in pixel coordinates using exclusive upper bounds.
                Valid coordinates lie within:
                    0 ≤ x_min ≤ x_max ≤ image width
                    0 ≤ y_min ≤ y_max ≤ image height
            - 'xyxy_normalized':
                Same as 'xyxy_absolute', but normalized to [0, 1] using:
                    x_normalized = x / image width
                    y_normalized = y / image height
                Coordinates follow the same exclusive convention.
            - 'xywh_absolute':
                [x, y, w, h] where (x, y) is the top-left corner in pixels and (w, h) are width and height.
                Conversion to 'xyxy_absolute' follows:
                    x_max = x + w
                    y_max = y + h
            - 'xywh_normalized':
                Same as 'xywh_absolute', but normalized to [0, 1] using image width and height.
            Defaults to 'xyxy_absolute'.
        target_format (str, optional):
            Desired output format. See `source_format`. Defaults to 'xywh_absolute'.
        image_size (tuple | list | None, optional):
            Required if converting to or from a normalized format. Should be (height, width).
            Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid. The structure matches the input structure.

    Returns:
        list: Converted bounding boxes in the target format. The structure matches the input structure.
    """
    # validate availability
    assert_log(expression=IMPORT_ERROR is None, message=f"Visual utilities are not available due to missing dependencies: {IMPORT_ERROR}")

    # parse arguments
    assert_type_value(obj=boxes, type_or_value=[list, tuple, np.ndarray], name="argument 'boxes'")
    assert_type_value(source_format, ["xyxy_absolute", "xyxy_normalized", "xywh_absolute", "xywh_normalized"], name="argument 'source_format'")
    assert_type_value(target_format, ["xyxy_absolute", "xyxy_normalized", "xywh_absolute", "xywh_normalized"], name="argument 'target_format'")
    assert_type_value(image_size, [tuple, list, None], name="argument 'image_size'")

    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    if len(boxes) == 0:
        return []

    if "normalized" in source_format or "normalized" in target_format:
        assert_log(expression=image_size is not None, message="Expected argument 'image_size' to be a tuple (height, width) when converting to or from normalized boxes.")
        assert_log(expression=len(image_size) == 2, message=f"Expected argument 'image_size' to be a 2-tuple (height, width) but got a tuple of length '{len(image_size)}'.")
        assert_log(expression=isinstance(image_size[0], int) and isinstance(image_size[1], int), message="Expected argument 'image_size' to be a 2-tuple (height, width) of integers.")
        assert_log(expression=image_size[0] > 0 and image_size[1] > 0, message="Expected argument 'image_size' to be a 2-tuple (height, width) of positive integers.")
        h, w = image_size

    # flatten input and store nested structure
    def flatten(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 4 and all(isinstance(e, (int, float, np.integer, np.floating)) and not isinstance(e, bool) for e in x):
                return [list(x)], None

            assert_log(expression=len(x) > 0, message="Expected argument 'boxes' to contain bounding boxes, but found an empty nested sequence.")

            flat = []
            shape = []
            for item in x:
                f, s = flatten(item)
                flat.extend(f)
                shape.append(s)
            return flat, shape

        assert_log(expression=False, message=f"Expected argument 'boxes' to contain only bounding boxes or nested sequences of bounding boxes, but got item of type '{type(x).__name__}'.")

    flat_boxes, structure = flatten(boxes)
    arr = np.asarray(flat_boxes, dtype=np.float64)

    # return early if formats match
    if source_format == target_format:
        if structure is None:
            return flat_boxes[0]

        flat_result = [box[:] for box in flat_boxes]

        def unflatten(s):
            if s is None:
                return flat_result.pop(0)
            return [unflatten(sub) for sub in s]

        return unflatten(structure)

    # convert to xyxy_absolute
    if source_format == "xyxy_absolute":
        tmp = arr
    elif source_format == "xyxy_normalized":
        tmp = arr * np.array([w, h, w, h], dtype=np.float64)
    elif source_format == "xywh_absolute":
        x, y, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        tmp = np.stack([x, y, x + bw, y + bh], axis=1)
    elif source_format == "xywh_normalized":
        x = arr[:, 0] * w
        y = arr[:, 1] * h
        bw = arr[:, 2] * w
        bh = arr[:, 3] * h
        tmp = np.stack([x, y, x + bw, y + bh], axis=1)
    else:
        raise NotImplementedError(f"Unknown source format '{source_format}'.")

    # convert from xyxy_absolute to target_format
    if target_format == "xyxy_absolute":
        result = np.round(tmp).astype(int)
        if image_size is not None:
            result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, w)
            result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, h)
    elif target_format == "xyxy_normalized":
        result = tmp / np.array([w, h, w, h], dtype=np.float64)
        result = np.clip(result, 0.0, 1.0)
    elif target_format == "xywh_absolute":
        xyxy = np.round(tmp).astype(int)
        if image_size is not None:
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        result = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    elif target_format == "xywh_normalized":
        x1, y1, x2, y2 = tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3]
        result = np.stack([x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h], axis=1)
        result = np.clip(result, 0.0, 1.0)
    else:
        raise NotImplementedError(f"Unknown target format '{target_format}'.")

    # restore original structure
    flat_result = result.tolist()
    if structure is None:
        return flat_result[0]

    def unflatten(s):
        if s is None:
            return flat_result.pop(0)
        return [unflatten(sub) for sub in s]

    result = unflatten(structure)
    return result

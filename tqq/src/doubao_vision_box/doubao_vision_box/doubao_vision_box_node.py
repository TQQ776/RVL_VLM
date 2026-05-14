import base64
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DoubaoVisionBoxNode(Node):
    """Send the latest camera frame to Doubao and draw returned boxes locally."""

    def __init__(self) -> None:
        super().__init__('doubao_vision_box')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('command_topic', '/doubao_vision_box/command')
        self.declare_parameter('annotated_topic', '/doubao_vision_box/annotated_image')
        self.declare_parameter('result_topic', '/doubao_vision_box/result_json')
        self.declare_parameter('model', 'doubao-seed-2-0-pro-260215')
        self.declare_parameter('base_url', 'https://ark.cn-beijing.volces.com/api/v3')
        self.declare_parameter('api_key_env', 'ARK_API_KEY')
        self.declare_parameter('timeout_sec', 90.0)
        self.declare_parameter('image_format', 'png')
        self.declare_parameter('vision_mode', 'auto')
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('max_image_width', 0)
        self.declare_parameter('max_cached_image_age_sec', 5.0)
        self.declare_parameter('output_dir', '/home/tqq/TQQ_ws/doubao_vision_outputs')
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'Doubao Vision Box')
        self.declare_parameter('save_images', True)
        self.declare_parameter('enable_stdin', False)
        self.declare_parameter('text_popup_enabled', True)
        self.declare_parameter('text_popup_key', 't')
        self.declare_parameter('text_popup_title', 'Doubao Vision Box')
        self.declare_parameter(
            'text_popup_prompt',
            '输入文本后按 Ctrl+Enter 发送，或按 Esc 取消。',
        )

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.command_topic = str(self.get_parameter('command_topic').value)
        self.annotated_topic = str(self.get_parameter('annotated_topic').value)
        self.result_topic = str(self.get_parameter('result_topic').value)
        self.model = str(self.get_parameter('model').value)
        self.base_url = str(self.get_parameter('base_url').value)
        self.api_key_env = str(self.get_parameter('api_key_env').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.image_format = str(self.get_parameter('image_format').value).strip().lower()
        self.vision_mode = str(self.get_parameter('vision_mode').value).strip().lower()
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.max_image_width = int(self.get_parameter('max_image_width').value)
        self.max_cached_image_age_sec = float(
            self.get_parameter('max_cached_image_age_sec').value
        )
        self.output_dir = str(self.get_parameter('output_dir').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.save_images = bool(self.get_parameter('save_images').value)
        self.enable_stdin = bool(self.get_parameter('enable_stdin').value)
        self.text_popup_enabled = bool(self.get_parameter('text_popup_enabled').value)
        self.text_popup_key = str(self.get_parameter('text_popup_key').value).strip().lower()
        self.text_popup_title = str(self.get_parameter('text_popup_title').value).strip()
        self.text_popup_prompt = str(self.get_parameter('text_popup_prompt').value).strip()

        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_stamp_sec: Optional[float] = None
        self.latest_encoding = ''
        self.image_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.keyboard_listener = None
        self.text_popup_lock = threading.Lock()
        self.text_popup_open = False
        self.text_popup_window = None
        self.display_lock = threading.Lock()
        self.display_image: Optional[np.ndarray] = None
        self.display_status = 'waiting for camera image...'
        self.display_hold_until = 0.0
        self.processing_request = False
        self.display_thread = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.create_subscription(String, self.command_topic, self.command_callback, 10)
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)
        self.result_pub = self.create_publisher(String, self.result_topic, 10)

        if self.show_window:
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()

        if self.enable_stdin:
            self.stdin_thread = threading.Thread(target=self.stdin_loop, daemon=True)
            self.stdin_thread.start()

        if self.text_popup_enabled:
            self._start_keyboard_listener()

        self.get_logger().info(
            'Doubao vision box ready. '
            f'image={self.image_topic}, command={self.command_topic}, '
                f'annotated={self.annotated_topic}, model={self.model}, mode={self.vision_mode}. '
            f'text_popup={self.text_popup_enabled}, text_key={self.text_popup_key}. '
            '按 t 弹出文本对话框，输入“你能看到什么”即可调用火山 Doubao 看当前相机图。'
        )

    def destroy_node(self) -> bool:
        self.shutdown_event.set()
        if self.display_thread is not None:
            self.display_thread.join(timeout=1.0)
        return super().destroy_node()

    def _display_loop(self) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            while rclpy.ok() and not self.shutdown_event.is_set():
                frame = self._display_frame()
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q')):
                    self.shutdown_event.set()
                    break
            cv2.destroyWindow(self.window_name)
        except Exception as exc:
            self.get_logger().error(f'Doubao vision display window failed: {exc}')

    def _display_frame(self) -> np.ndarray:
        now = time.time()
        with self.image_lock:
            latest = None if self.latest_image is None else self.latest_image.copy()
        with self.display_lock:
            display = None if self.display_image is None else self.display_image.copy()
            status = self.display_status
            hold_active = now < self.display_hold_until
            processing = self.processing_request

        if display is not None and hold_active:
            frame = display
        elif latest is not None:
            frame = latest
            status = 'camera preview' if not processing else status
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if status or processing:
            overlay_text = status or 'processing...'
            self._draw_status_bar(frame, overlay_text)
        return frame

    def _set_display_status(self, status: str) -> None:
        with self.display_lock:
            self.display_status = status

    def _set_display_result(self, image: np.ndarray, status: str, hold_sec: float = 60.0) -> None:
        with self.display_lock:
            self.display_image = image.copy()
            self.display_status = status
            self.display_hold_until = time.time() + hold_sec

    def _set_processing_request(self, is_processing: bool, status: str = '') -> None:
        with self.display_lock:
            self.processing_request = is_processing
            if status:
                self.display_status = status

    def _draw_status_bar(self, image: np.ndarray, text: str) -> None:
        if image.size == 0:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        text = str(text)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 8
        cv2.rectangle(
            image,
            (0, 0),
            (min(image.shape[1], tw + pad * 2), th + baseline + pad * 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (pad, th + pad),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _start_keyboard_listener(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:
            self.get_logger().error(
                'text popup is enabled, but pynput is not available or cannot connect '
                'to the desktop keyboard event source. Install it with: '
                f'/usr/bin/python3 -m pip install --user -U pynput. Details: {exc}'
            )
            return

        try:
            self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
            self.get_logger().info(f'text popup ready: press {self.text_popup_key}')
        except Exception as exc:
            self.get_logger().error(f'Failed to start text popup keyboard listener: {exc}')

    def _key_matches_name(self, key, wanted: str) -> bool:
        wanted = wanted.lower()
        char = getattr(key, 'char', None)
        if char is not None:
            return str(char).lower() == wanted
        key_name = getattr(key, 'name', None)
        if key_name is None:
            key_name = str(key).replace('Key.', '')
        return str(key_name).lower() == wanted

    def _on_key_press(self, key) -> None:
        if not self.text_popup_enabled:
            return
        if self._text_popup_is_open():
            return
        if self._key_matches_name(key, self.text_popup_key):
            self._open_text_popup_from_key()

    def _text_popup_is_open(self) -> bool:
        with self.text_popup_lock:
            return self.text_popup_open

    def _set_text_popup_open(self, is_open: bool) -> None:
        with self.text_popup_lock:
            self.text_popup_open = is_open

    def _open_text_popup_from_key(self) -> None:
        with self.text_popup_lock:
            if self.text_popup_open:
                window = self.text_popup_window
                if window is not None:
                    window.bring_to_front()
                return
            self.text_popup_open = True

        threading.Thread(target=self._text_popup_worker, daemon=True).start()

    def _text_popup_worker(self) -> None:
        try:
            window = TextConversationWindow(
                title=self.text_popup_title or 'Doubao Vision Box',
                prompt=self.text_popup_prompt,
                submit_callback=self._handle_text_popup_submit,
                close_callback=lambda: self._set_text_popup_open(False),
            )
            with self.text_popup_lock:
                self.text_popup_window = window
            window.run()
        except Exception as exc:
            self.get_logger().error(f'Text popup failed: {exc}')
        finally:
            with self.text_popup_lock:
                self.text_popup_window = None
                self.text_popup_open = False

    def _handle_text_popup_submit(self, text: str) -> str:
        if not text:
            return ''
        ok, message = self.handle_command(text)
        return message if message else ('处理完成。' if ok else '处理失败。')

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(f'Could not convert image: {exc}')
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        with self.image_lock:
            self.latest_image = image.copy()
            self.latest_stamp_sec = stamp
            self.latest_encoding = msg.encoding
        self._set_display_status('camera preview')

    def command_callback(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        threading.Thread(target=self.handle_command, args=(text,), daemon=True).start()

    def stdin_loop(self) -> None:
        while rclpy.ok() and not self.shutdown_event.is_set():
            try:
                text = input('Doubao> ').strip()
            except EOFError:
                return
            except Exception as exc:
                self.get_logger().warn(f'stdin read failed: {exc}')
                time.sleep(0.2)
                continue
            if not text:
                continue
            threading.Thread(target=self.handle_command, args=(text,), daemon=True).start()

    def handle_command(self, text: str) -> Tuple[bool, str]:
        if not self._is_vision_box_request(text):
            message = (
                '这个独立节点只处理看图、找物体、框选标注类请求。'
                '例如：你能看到什么？请把图中的橘子框出来。'
            )
            self.get_logger().info(f'Ignored command: {text}. {message}')
            return False, message
        if not self.request_lock.acquire(blocking=False):
            message = '正在处理上一条 Doubao 看图请求，请稍等。'
            self.get_logger().warn(message)
            return False, message
        try:
            self._set_processing_request(True, 'sending image to Doubao...')
            image, age = self._latest_image_copy()
            self.get_logger().info(
                f'Doubao vision request: "{text}", image={image.shape[1]}x{image.shape[0]}, '
                f'age={age:.2f}s'
            )
            result_image, answer, boxes, mode_used = self._call_doubao_vision(text, image)
            output_path = ''
            if mode_used != 'direct':
                output_path = self._save_outputs_if_needed(image, result_image, answer, boxes)
            self._publish_result(text, answer, output_path, image.shape[1], image.shape[0], boxes, mode_used)
            if mode_used != 'direct':
                self._publish_annotated(result_image)
            if self.show_window:
                if mode_used == 'direct':
                    self._set_display_status('Doubao answered')
                else:
                    self._set_display_result(
                        result_image,
                        'Doubao returned vision result',
                        hold_sec=60.0,
                    )
            answer_text = self._answer_summary(answer, output_path, boxes, mode_used)
            self.get_logger().info(
                f'Doubao returned vision result, mode={mode_used}, boxes={len(boxes)}. '
                f'output={output_path or "not saved"}'
            )
            return True, answer_text
        except Exception as exc:
            self.get_logger().error(f'Doubao vision box failed: {exc}')
            self._publish_error(text, str(exc))
            self._set_display_status(f'Doubao failed: {exc}')
            return False, f'处理失败：{exc}'
        finally:
            self._set_processing_request(False)
            self.request_lock.release()

    def _answer_summary(
        self,
        answer: str,
        output_path: str,
        boxes: List[Dict],
        mode_used: str,
    ) -> str:
        box_text = ''
        if boxes:
            lines = [
                f'box{i + 1}: xmin={box["xmin"]}, ymin={box["ymin"]}, '
                f'xmax={box["xmax"]}, ymax={box["ymax"]}'
                for i, box in enumerate(boxes)
            ]
            box_text = '\n' + '\n'.join(lines)
        suffix = f'\n已保存：{output_path}' if output_path else ''
        answer_line = self._human_answer_from_model_response(answer)
        if mode_used == 'box_percent':
            prefix = answer_line or '已根据 API 返回坐标在原图上绘制标注。'
        else:
            prefix = answer_line or str(answer or '').strip() or '已返回 Doubao 视觉回答。'
        return f'{prefix}{box_text}{suffix}'

    def _wants_box_selection(self, text: str) -> bool:
        lowered = text.lower()
        edit_triggers = [
            '框',
            '框出',
            '框住',
            '圈出',
            '圈住',
            '标注',
            '画框',
            '用框',
            'box',
            'bounding',
            'bbox',
        ]
        return any(trigger in lowered for trigger in edit_triggers)

    def _is_vision_box_request(self, text: str) -> bool:
        lowered = text.lower()
        triggers = [
            '看到什么',
            '看见什么',
            '有什么',
            '图中',
            '图像',
            '图片',
            '相机',
            '画面',
            '找出',
            '框选',
            '框出',
            '框住',
            '画框',
            '圈出',
            '圈住',
            '标注',
            'box',
            'bounding',
        ]
        return any(trigger in lowered for trigger in triggers)

    def _latest_image_copy(self) -> Tuple[np.ndarray, float]:
        with self.image_lock:
            if self.latest_image is None or self.latest_stamp_sec is None:
                raise RuntimeError(f'No image received yet on {self.image_topic}.')
            image = self.latest_image.copy()
            age = time.time() - self.latest_stamp_sec
        if age > self.max_cached_image_age_sec:
            raise RuntimeError(
                f'Latest image is stale: {age:.1f}s old on {self.image_topic}.'
            )
        return image, age

    def _call_doubao_vision(
        self,
        question: str,
        bgr: np.ndarray,
    ) -> Tuple[np.ndarray, str, List[Dict], str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f'Environment variable {self.api_key_env} is empty. '
                f'Run: export {self.api_key_env}=你的火山引擎API_KEY'
            )

        mode = self.vision_mode
        if mode == 'auto':
            mode = 'box_percent' if self._wants_box_selection(question) else 'direct'

        if mode in ('direct', 'seed', 'vision', 'text_image'):
            answer = self._call_doubao_direct_vision(question, bgr, api_key)
            return bgr.copy(), answer, [], 'direct'

        boxes, answer = self._call_doubao_percent_boxes(question, bgr, api_key)
        result_bgr = self._draw_boxes_on_original_image(bgr, boxes)
        return result_bgr, answer, boxes, 'box_percent'

    def _call_doubao_direct_vision(
        self,
        question: str,
        bgr: np.ndarray,
        api_key: str,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        image_url, _, _ = self._image_to_data_url(bgr)
        prompt = self._build_direct_vision_prompt(question)
        client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout_sec)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': (
                                '你是一个正常对话的视觉助手。只输出最终回答，'
                                '绝对不要输出思考过程、内心独白、推理草稿、'
                                '“首先看看”“用户问”“我需要”等分析性文字。'
                            ),
                        }
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_image', 'image_url': image_url},
                        {'type': 'input_text', 'text': prompt},
                    ],
                }
            ],
        )
        answer_text = self._responses_output_to_text(response)
        return answer_text

    def _call_doubao_percent_boxes(
        self,
        question: str,
        bgr: np.ndarray,
        api_key: str,
    ) -> Tuple[List[Dict], str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        image_url, _, _ = self._image_to_data_url(bgr)
        h, w = bgr.shape[:2]
        labels = self._target_labels_from_text(question)
        prompt = self._build_percent_box_prompt(question, labels, w, h)
        client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout_sec)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': (
                                '你是一个视觉定位助手。只返回严格 JSON，不要 Markdown，'
                                '不要解释，不要输出思考过程，不要返回图片，不要生成标注图。'
                                '所有标签必须使用英文。'
                            ),
                        }
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_image', 'image_url': image_url},
                        {'type': 'input_text', 'text': prompt},
                    ],
                }
            ],
        )
        answer_text = self._responses_output_to_text(response)
        boxes = self._parse_api_boxes(answer_text, w, h)
        if labels != ['object']:
            self._apply_labels_to_boxes(boxes, labels)
        boxes = self._dedupe_boxes(boxes)
        return boxes, answer_text

    def _build_percent_box_prompt(
        self,
        question: str,
        labels: List[str],
        width: int,
        height: int,
    ) -> str:
        labels = [label for label in labels if label and label != 'object']
        if self._requests_all_boxes(question):
            target_hint = (
                '用户要求框选多个/全部可见目标。请为每一个清晰可见且可命名的目标分别返回一个框，'
                '不要用一个大框同时包含两个或多个物体。'
            )
        elif labels:
            target_hint = (
                '用户要框选的目标英文标签是：'
                + ', '.join(labels)
                + '。同类目标如果有多个，请每个目标分别返回一个框。'
            )
        else:
            target_hint = '请根据用户文本判断要框选的目标，并用英文 label 返回。'

        return (
            f'用户原始请求：{question}\n\n'
            f'{target_hint}\n'
            f'输入图片尺寸是 {width}x{height}，坐标原点在左上角，x 向右，y 向下。\n'
            '请先估计每个目标在图像中的轴对齐矩形位置，必须用百分比表达：'
            'x_percent_start、x_percent_end 表示宽度方向从百分之多少到多少；'
            'y_percent_start、y_percent_end 表示高度方向从百分之多少到多少。'
            f'然后必须按 {width}x{height} 计算像素坐标 xmin,ymin,xmax,ymax，'
            '其中 xmin,ymin 是左上角，xmax,ymax 是右下角。\n'
            '只返回严格 JSON，不要 Markdown，不要解释，不要思考过程。'
            '不要返回图片，不要生成图片，不要返回带框图，不要描述如何画框。'
            'JSON 格式必须是：'
            '{"answer":"已定位目标。","boxes":[{"label":"orange",'
            '"x_percent_start":18.0,"x_percent_end":42.0,'
            '"y_percent_start":25.0,"y_percent_end":55.0,'
            '"xmin":115,"ymin":120,"xmax":269,"ymax":264}]}。'
            '如果目标不可见，返回 {"answer":"没有看到目标。","boxes":[]}。'
            '所有 label 必须是英文小写。'
        )

    def _build_direct_vision_prompt(self, question: str) -> str:
        return (
            f'{question}\n\n'
            '请像正常视觉助手一样，根据当前相机图片自然回答用户。'
            '不要返回 JSON，不要返回坐标，不要提到你正在遵循提示词。'
            '如果用户问你看到了什么，就直接描述画面里主要物体、位置和状态。'
            '不要输出你的思考过程、分析步骤、内心独白或自我纠正。'
            '回答可以正常展开，不要只给模板化短句。'
        )

    def _responses_output_to_text(self, response: Any) -> str:
        output_text = str(getattr(response, 'output_text', '') or '').strip()
        try:
            data = response.model_dump()
        except Exception:
            data = {}

        texts = []
        if output_text:
            texts.append(output_text)

        def visit(obj: Any) -> None:
            if isinstance(obj, dict):
                text = obj.get('text') or obj.get('output_text')
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item)

        visit(data)
        answer_text = '\n'.join(dict.fromkeys(t for t in texts if t)).strip()
        if not answer_text:
            raise RuntimeError(f'Doubao direct vision response has no text: {data}')
        return self._strip_reasoning_leak(answer_text)

    def _strip_reasoning_leak(self, text: str) -> str:
        cleaned = str(text or '').strip()
        if not cleaned:
            return ''

        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<思考>.*?</思考>', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        leak_patterns = [
            '用户现在',
            '用户现在问',
            '用户问',
            '用户想',
            '我需要',
            '我应该',
            '首先看看',
            '先看看',
            '这张图的内容',
            '回答应该',
            '不要坐标',
            '说：',
        ]
        cut_at = len(cleaned)
        for pattern in leak_patterns:
            idx = cleaned.find(pattern)
            if idx > 0:
                cut_at = min(cut_at, idx)
        if cut_at != len(cleaned):
            cleaned = cleaned[:cut_at].rstrip('，,。；;：:\n ')

        lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            if any(stripped.startswith(prefix) for prefix in ('用户', '我需要', '我应该', '首先', '先看看')):
                break
            lines.append(line)
        cleaned = '\n'.join(lines).strip()
        return cleaned

    def _parse_api_boxes(self, text: str, width: int, height: int) -> List[Dict]:
        try:
            payload = self._extract_json(text)
        except Exception as exc:
            self.get_logger().debug(f'Could not parse API box JSON: {exc}; text={text[:500]}')
            return []
        boxes_raw = payload.get('boxes') if isinstance(payload, dict) else payload
        if not isinstance(boxes_raw, list):
            return []

        boxes = []
        for item in boxes_raw:
            if not isinstance(item, dict):
                continue
            try:
                xmin, ymin, xmax, ymax, source = self._box_coordinates_from_item(
                    item,
                    width,
                    height,
                )
            except (KeyError, TypeError, ValueError):
                continue
            xmin = int(np.clip(xmin, 0, width - 1))
            ymin = int(np.clip(ymin, 0, height - 1))
            xmax = int(np.clip(xmax, xmin + 1, width))
            ymax = int(np.clip(ymax, ymin + 1, height))
            boxes.append(
                {
                    'label': self._english_label(item.get('label') or item.get('label_zh') or 'object'),
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'x_percent_start': self._optional_float(item, 'x_percent_start'),
                    'x_percent_end': self._optional_float(item, 'x_percent_end'),
                    'y_percent_start': self._optional_float(item, 'y_percent_start'),
                    'y_percent_end': self._optional_float(item, 'y_percent_end'),
                    'source': source,
                }
            )
        return boxes

    def _box_coordinates_from_item(
        self,
        item: Dict,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int, str]:
        if all(key in item for key in ('xmin', 'ymin', 'xmax', 'ymax')):
            return (
                int(round(float(item['xmin']))),
                int(round(float(item['ymin']))),
                int(round(float(item['xmax']))),
                int(round(float(item['ymax']))),
                'api_response_xyxy',
            )
        if all(key in item for key in ('x1', 'y1', 'x2', 'y2')):
            return (
                int(round(float(item['x1']))),
                int(round(float(item['y1']))),
                int(round(float(item['x2']))),
                int(round(float(item['y2']))),
                'api_response_x1y1x2y2',
            )

        x_start = self._first_number(
            item,
            ('x_percent_start', 'x_start_percent', 'width_percent_start', 'left_percent'),
        )
        x_end = self._first_number(
            item,
            ('x_percent_end', 'x_end_percent', 'width_percent_end', 'right_percent'),
        )
        y_start = self._first_number(
            item,
            ('y_percent_start', 'y_start_percent', 'height_percent_start', 'top_percent'),
        )
        y_end = self._first_number(
            item,
            ('y_percent_end', 'y_end_percent', 'height_percent_end', 'bottom_percent'),
        )
        if None in (x_start, x_end, y_start, y_end):
            raise KeyError('box item does not contain xyxy or percent fields')
        return (
            int(round(float(x_start) / 100.0 * width)),
            int(round(float(y_start) / 100.0 * height)),
            int(round(float(x_end) / 100.0 * width)),
            int(round(float(y_end) / 100.0 * height)),
            'api_response_percent',
        )

    def _first_number(self, item: Dict, keys: Tuple[str, ...]) -> Optional[float]:
        for key in keys:
            value = self._optional_float(item, key)
            if value is not None:
                return value
        return None

    def _optional_float(self, item: Dict, key: str) -> Optional[float]:
        try:
            value = item.get(key)
        except AttributeError:
            return None
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _human_answer_from_model_response(self, text: str) -> str:
        try:
            payload = self._extract_json(text)
        except Exception:
            cleaned = str(text or '').strip()
            return cleaned[:300] if cleaned else ''
        if isinstance(payload, dict):
            answer = str(payload.get('answer') or payload.get('text') or '').strip()
            return answer[:300]
        return ''

    def _draw_boxes(self, image: np.ndarray, boxes: List[Dict]) -> np.ndarray:
        canvas = image.copy()
        for box in boxes:
            x1 = int(box['xmin'])
            y1 = int(box['ymin'])
            x2 = int(box['xmax'])
            y2 = int(box['ymax'])
            label = str(box.get('label') or 'object')
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_boxes_on_original_image(self, raw_bgr: np.ndarray, boxes: List[Dict]) -> np.ndarray:
        if boxes:
            return self._draw_boxes(raw_bgr, boxes)
        return raw_bgr.copy()

    def _target_labels_from_text(self, text: str) -> List[str]:
        lowered = str(text or '').lower()
        label_map = [
            (('橘子', '橙子', 'orange'), 'orange'),
            (('香蕉', 'banana'), 'banana'),
            (('苹果', 'apple'), 'apple'),
            (('茄子', 'eggplant', 'aubergine'), 'eggplant'),
            (('萝卜', '白萝卜', 'radish', 'daikon'), 'radish'),
            (('胡萝卜', 'carrot'), 'carrot'),
            (('梨', 'pear'), 'pear'),
            (('杯子', '水杯', 'cup'), 'cup'),
            (('瓶子', 'bottle'), 'bottle'),
            (('碗', 'bowl'), 'bowl'),
            (('盘子', 'plate'), 'plate'),
            (('鼠标', 'mouse'), 'mouse'),
            (('键盘', 'keyboard'), 'keyboard'),
            (('手机', 'phone'), 'phone'),
        ]
        found = []
        for aliases, label in label_map:
            positions = [lowered.find(alias) for alias in aliases if alias in lowered]
            if positions:
                found.append((min(positions), label))
        for match in re.finditer(
            r'\b(?:box|bbox|mark|label)\s+([a-z][a-z0-9_-]{1,30})\b',
            lowered,
        ):
            found.append((match.start(1), match.group(1)))
        found.sort(key=lambda item: item[0])
        labels = []
        for _, label in found:
            if label not in labels:
                labels.append(label)
        return labels or ['object']

    def _english_label(self, label: Any) -> str:
        raw = str(label or '').strip().lower()
        if not raw:
            return 'object'
        label_map = {
            '橘子': 'orange',
            '橙子': 'orange',
            'orange': 'orange',
            '香蕉': 'banana',
            'banana': 'banana',
            '苹果': 'apple',
            'apple': 'apple',
            '茄子': 'eggplant',
            'eggplant': 'eggplant',
            'aubergine': 'eggplant',
            '萝卜': 'radish',
            '白萝卜': 'radish',
            'radish': 'radish',
            'daikon': 'radish',
            '胡萝卜': 'carrot',
            'carrot': 'carrot',
            '梨': 'pear',
            'pear': 'pear',
            '杯子': 'cup',
            '水杯': 'cup',
            'cup': 'cup',
            '瓶子': 'bottle',
            'bottle': 'bottle',
            '碗': 'bowl',
            'bowl': 'bowl',
            '盘子': 'plate',
            'plate': 'plate',
            '鼠标': 'mouse',
            'mouse': 'mouse',
            '键盘': 'keyboard',
            'keyboard': 'keyboard',
            '手机': 'phone',
            'phone': 'phone',
            '物体': 'object',
            '目标': 'object',
            'object': 'object',
        }
        return label_map.get(raw, re.sub(r'[^a-z0-9_-]+', '_', raw).strip('_') or 'object')

    def _requests_all_boxes(self, text: str) -> bool:
        lowered = str(text or '').lower()
        triggers = [
            '所有',
            '全部',
            '每个',
            '每一个',
            '全部物体',
            '所有物体',
            '多个物体',
            '所有东西',
            'all',
            'everything',
            'objects',
        ]
        return any(trigger in lowered for trigger in triggers)

    def _apply_labels_to_boxes(self, boxes: List[Dict], labels: List[str]) -> None:
        labels = [self._english_label(label) for label in labels if str(label or '').strip()]
        if not labels:
            labels = ['object']
        for index, box in enumerate(boxes):
            current_label = self._english_label(box.get('label'))
            if len(labels) == 1 or current_label == 'object':
                box['label'] = labels[min(index, len(labels) - 1)]
            else:
                box['label'] = current_label

    def _parse_bbox_string(self, bbox_str: str, width: int, height: int) -> List[Dict]:
        text = str(bbox_str or '').strip()
        if not text:
            return []

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'(\[.*\]|\{.*\})', text, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    parsed = None

        raw_boxes = []
        if parsed is not None:
            raw_boxes = self._boxes_from_parsed_bbox_payload(parsed)
        if not raw_boxes:
            numbers = [float(value) for value in re.findall(r'-?\d+(?:\.\d+)?', text)]
            raw_boxes = [numbers[i : i + 4] for i in range(0, len(numbers) - 3, 4)]

        boxes = []
        for raw in raw_boxes:
            if len(raw) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in raw[:4]]
            except (TypeError, ValueError):
                continue
            boxes.append(self._normalize_box_to_xyxy(x1, y1, x2, y2, width, height))
        return [box for box in boxes if box is not None]

    def _boxes_from_parsed_bbox_payload(self, payload: Any) -> List[List[float]]:
        if isinstance(payload, dict):
            if all(key in payload for key in ('xmin', 'ymin', 'xmax', 'ymax')):
                return [[payload['xmin'], payload['ymin'], payload['xmax'], payload['ymax']]]
            if all(key in payload for key in ('x1', 'y1', 'x2', 'y2')):
                return [[payload['x1'], payload['y1'], payload['x2'], payload['y2']]]
            boxes = payload.get('boxes') or payload.get('bboxes') or payload.get('bbox')
            return self._boxes_from_parsed_bbox_payload(boxes)
        if isinstance(payload, list):
            if len(payload) >= 4 and all(isinstance(v, (int, float, str)) for v in payload[:4]):
                return [payload[:4]]
            boxes = []
            for item in payload:
                boxes.extend(self._boxes_from_parsed_bbox_payload(item))
            return boxes
        return []

    def _normalize_box_to_xyxy(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> Optional[Dict]:
        max_value = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if max_value <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            source = 'api_response_normalized_0_1'
        elif max_value <= 1000.0 and (max(x1, x2) > width or max(y1, y2) > height):
            x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
            y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
            source = 'api_response_normalized_1000'
        else:
            source = 'api_response'

        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        xmin = int(np.clip(round(xmin), 0, width - 1))
        ymin = int(np.clip(round(ymin), 0, height - 1))
        xmax = int(np.clip(round(xmax), xmin + 1, width))
        ymax = int(np.clip(round(ymax), ymin + 1, height))
        if xmax <= xmin or ymax <= ymin:
            return None
        return {
            'label': 'object',
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'source': source,
        }

    def _dedupe_boxes(self, boxes: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for box in boxes:
            key = (box.get('xmin'), box.get('ymin'), box.get('xmax'), box.get('ymax'))
            if key in seen:
                continue
            seen.add(key)
            unique.append(box)
        return unique

    def _extract_json(self, text: str) -> Any:
        text = str(text or '').strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r'(\{.*\}|\[.*\])', text, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f'API response is not JSON: {text[:500]}')
        return json.loads(match.group(1))

    def _image_to_data_url(self, bgr: np.ndarray) -> Tuple[str, int, int]:
        image = bgr
        h, w = image.shape[:2]
        send_w, send_h = w, h
        if self.max_image_width > 0 and w > self.max_image_width:
            scale = self.max_image_width / float(w)
            send_w = self.max_image_width
            send_h = max(1, int(round(h * scale)))
            image = cv2.resize(image, (send_w, send_h), interpolation=cv2.INTER_AREA)

        if self.image_format == 'jpg':
            self.image_format = 'jpeg'
        if self.image_format not in ('png', 'jpeg'):
            raise RuntimeError(f'Unsupported image_format: {self.image_format}. Use png or jpeg.')

        ext = '.png' if self.image_format == 'png' else '.jpg'
        encode_params = []
        if self.image_format == 'jpeg':
            quality = int(np.clip(self.jpeg_quality, 1, 100))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, encoded = cv2.imencode(ext, image, encode_params)
        if not ok:
            raise RuntimeError(f'Failed to encode camera image as {self.image_format.upper()}.')
        b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
        return f'data:image/{self.image_format};base64,{b64}', send_w, send_h

    def _save_outputs_if_needed(
        self,
        raw: np.ndarray,
        annotated: np.ndarray,
        answer: str,
        boxes: List[Dict],
    ) -> str:
        if not self.save_images:
            return ''
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        raw_path = os.path.join(self.output_dir, f'{stamp}_raw.jpg')
        annotated_path = os.path.join(self.output_dir, f'{stamp}_annotated.jpg')
        json_path = os.path.join(self.output_dir, f'{stamp}_result.json')
        cv2.imwrite(raw_path, raw)
        cv2.imwrite(annotated_path, annotated)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'raw_image': raw_path,
                    'annotated_image': annotated_path,
                    'boxes_xyxy': boxes,
                    'model_response': answer,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return annotated_path

    def _publish_annotated(self, annotated: np.ndarray) -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(f'Could not publish annotated image: {exc}')
            return
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'doubao_vision_box'
        self.annotated_pub.publish(msg)

    def _publish_result(
        self,
        command: str,
        answer: str,
        output_path: str,
        width: int,
        height: int,
        boxes: List[Dict],
        mode_used: str,
    ) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                'ok': True,
                'mode': mode_used,
                'command': command,
                'model': self.model,
                'image_width': width,
                'image_height': height,
                'boxes_xyxy': boxes,
                'output_image': output_path,
                'model_response': answer,
            },
            ensure_ascii=False,
        )
        self.result_pub.publish(msg)

    def _publish_error(self, command: str, error: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {'ok': False, 'command': command, 'error': error},
            ensure_ascii=False,
        )
        self.result_pub.publish(msg)


class TextConversationWindow:
    """Tk popup that keeps a small text chat history."""

    def __init__(self, title: str, prompt: str, submit_callback, close_callback) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError(
                'tkinter is not installed. Install it with: sudo apt install python3-tk'
            ) from exc

        self.tk = tk
        self.ttk = ttk
        self.submit_callback = submit_callback
        self.close_callback = close_callback
        self.result_queue = queue.Queue()
        self.processing = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('760x560')
        self.root.minsize(560, 360)

        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill='both', expand=True)

        prompt_label = ttk.Label(frame, text=prompt, anchor='w')
        prompt_label.pack(fill='x', pady=(0, 8))

        self.history = tk.Text(frame, wrap='word', height=16, state='disabled')
        self.history.pack(fill='both', expand=True)
        self._configure_tags()

        input_frame = ttk.Frame(frame)
        input_frame.pack(fill='x', pady=(10, 0))

        self.input_text = tk.Text(input_frame, wrap='word', height=4)
        self.input_text.pack(fill='x', expand=False)

        buttons = ttk.Frame(frame)
        buttons.pack(fill='x', pady=(8, 0))

        self.status_var = tk.StringVar(value='Ctrl+Enter 发送，Esc 关闭窗口。')
        status = ttk.Label(buttons, textvariable=self.status_var, anchor='w')
        status.pack(side='left', fill='x', expand=True)

        self.send_button = ttk.Button(buttons, text='发送 Ctrl+Enter', command=self.submit)
        self.send_button.pack(side='right')
        self.close_button = ttk.Button(buttons, text='关闭 Esc', command=self.close)
        self.close_button.pack(side='right', padx=(0, 8))

        self.root.bind('<Control-Return>', self.submit)
        self.root.bind('<Escape>', self.close)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.input_text.focus_set()
        self.root.after(100, self.root.lift)
        self.root.after(100, self._poll_results)

    def _configure_tags(self) -> None:
        self.history.tag_configure(
            'user_name',
            foreground='#0b5cad',
            font=('TkDefaultFont', 10, 'bold'),
        )
        self.history.tag_configure(
            'assistant_name',
            foreground='#126b35',
            font=('TkDefaultFont', 10, 'bold'),
        )

    def run(self) -> None:
        self.root.mainloop()

    def bring_to_front(self) -> None:
        self.root.after(0, self._bring_to_front)

    def _bring_to_front(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.input_text.focus_set()

    def submit(self, event=None):
        if self.processing:
            return 'break'
        text = self.input_text.get('1.0', 'end').strip()
        if not text:
            return 'break'

        self.input_text.delete('1.0', 'end')
        self._append_message('我', text, name_tag='user_name')
        self._set_processing(True)

        threading.Thread(target=self._submit_worker, args=(text,), daemon=True).start()
        return 'break'

    def _submit_worker(self, text: str) -> None:
        try:
            answer = str(self.submit_callback(text) or '').strip()
        except Exception as exc:
            answer = f'处理失败：{exc}'
        self.result_queue.put(answer)

    def _poll_results(self) -> None:
        try:
            while True:
                answer = self.result_queue.get_nowait()
                if answer:
                    self._append_message('Doubao', answer, name_tag='assistant_name')
                self._set_processing(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _append_message(self, speaker: str, text: str, name_tag: str) -> None:
        self.history.configure(state='normal')
        if self.history.index('end-1c') != '1.0':
            self.history.insert('end', '\n\n')
        self.history.insert('end', f'{speaker}：', name_tag)
        self.history.insert('end', f'\n{text}')
        self.history.configure(state='disabled')
        self.history.see('end')

    def _set_processing(self, is_processing: bool) -> None:
        self.processing = is_processing
        state = 'disabled' if is_processing else 'normal'
        self.send_button.configure(state=state)
        self.status_var.set('正在处理，请稍等...' if is_processing else 'Ctrl+Enter 发送，Esc 关闭窗口。')

    def close(self, event=None):
        if self.processing:
            self.status_var.set('正在处理当前消息，完成后再关闭。')
            return 'break'
        try:
            self.close_callback()
        finally:
            self.root.destroy()
        return 'break'


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DoubaoVisionBoxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

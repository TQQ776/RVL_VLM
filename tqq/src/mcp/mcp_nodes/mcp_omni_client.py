import base64
import io
import json
import os
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from mcp.srv import MoveAxis, ObjectName
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from speech.whisper_speak import WhisperSpeak
from std_msgs.msg import String
from std_srvs.srv import Trigger


class McpOmniClient(WhisperSpeak):
    """Qwen3.5-Omni Realtime multimodal MCP client."""

    def __init__(self) -> None:
        super().__init__('mcp_omni_client')

        self._latest_vision_image = None
        self._latest_vision_image_time = 0.0
        self._vision_image_lock = threading.Lock()
        self._latest_yolo_objects_payload = None
        self._latest_yolo_objects_time = 0.0
        self._yolo_objects_lock = threading.Lock()
        self._latest_api_detection_image_header = None
        self._vision_window_shutdown = threading.Event()
        self._vision_display_lock = threading.Lock()
        self._vision_display_image = None
        self._vision_latest_display_image = None
        self._vision_display_status = 'waiting for camera image...'
        self._vision_display_hold_until = 0.0
        self._vision_display_thread = None
        self._last_api_detection_raw_json = {}

        self.api_detections_pub = self.create_publisher(String, self.api_detections_topic, 10)
        self.target_command_pub = self.create_publisher(String, self.target_command_topic, 10)

        self.go_home_client = self.create_client(Trigger, self.go_home_service)
        self.move_x_client = self.create_client(MoveAxis, self.move_x_service)
        self.move_y_client = self.create_client(MoveAxis, self.move_y_service)
        self.move_z_client = self.create_client(MoveAxis, self.move_z_service)
        self.open_gripper_client = self.create_client(Trigger, self.open_gripper_service)
        self.close_gripper_client = self.create_client(Trigger, self.close_gripper_service)
        self.list_yolo_objects_client = self.create_client(
            Trigger,
            self.list_yolo_objects_service,
        )
        self.grab_object_client = self.create_client(
            ObjectName,
            self.grab_object_service,
        )
        self._yolo_objects_sub = self.create_subscription(
            String,
            self.detected_objects_topic,
            self._yolo_objects_callback,
            10,
        )

        self._vision_image_sub = None
        if self.vision_enabled:
            image_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._vision_image_sub = self.create_subscription(
                RosImage,
                self.vision_image_topic,
                self._vision_image_callback,
                image_qos,
            )
            if self.vision_show_window:
                self._vision_display_thread = threading.Thread(
                    target=self._vision_display_loop,
                    daemon=True,
                )
                self._vision_display_thread.start()

        self.get_logger().info(
            'Qwen-Omni Realtime MCP client ready. '
            f'model={self.omni_model}, voice={self.omni_realtime_voice}, '
            f'realtime_url={self.omni_realtime_url}, vision_topic={self.vision_image_topic}, '
            f'api_detections={self.api_detections_topic}, '
            f'vision_window={self.vision_show_window}, save_images={self.vision_save_images}.'
        )

    def destroy_node(self) -> bool:
        self._vision_window_shutdown.set()
        if self._vision_display_thread is not None:
            self._vision_display_thread.join(timeout=1.0)
        return super().destroy_node()

    def _declare_parameters(self) -> None:
        super()._declare_parameters()
        self.declare_parameter('omni_api_key_env', 'DASHSCOPE_API_KEY')
        self.declare_parameter('omni_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.declare_parameter('omni_model', 'qwen3.5-omni-plus-realtime')
        self.declare_parameter('omni_text_model', 'qwen3.5-omni-plus')
        self.declare_parameter('omni_realtime_url', 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime')
        self.declare_parameter('omni_realtime_voice', 'Ethan')
        self.declare_parameter('omni_realtime_chunk_bytes', 3200)
        self.declare_parameter('omni_realtime_chunk_sleep_sec', 0.01)
        self.declare_parameter('omni_realtime_enable_search', False)
        self.declare_parameter('omni_realtime_connect_retries', 3)
        self.declare_parameter('omni_realtime_connect_retry_delay_sec', 1.0)
        self.declare_parameter('omni_timeout', 90.0)
        self.declare_parameter('omni_max_tokens', 1000)
        self.declare_parameter('omni_native_audio_output_enabled', True)
        self.declare_parameter('omni_native_audio_fallback_to_local_tts', False)
        self.declare_parameter('omni_native_audio_sample_rate', 24000)
        self.declare_parameter('omni_native_audio_silence_bytes', 3200)
        self.declare_parameter('omni_native_audio_timeout', 90.0)
        self.declare_parameter('omni_speech_rate', 'normal')
        self.declare_parameter('omni_speech_volume', 'normal')
        self.declare_parameter('omni_speech_emotion', 'natural')
        self.declare_parameter('omni_speech_style', '清晰、自然、友好，适合机器人语音助手')

        self.declare_parameter('go_home_service', '/mcp_server/go_home')
        self.declare_parameter('move_x_service', '/mcp_server/move_x_cm')
        self.declare_parameter('move_y_service', '/mcp_server/move_y_cm')
        self.declare_parameter('move_z_service', '/mcp_server/move_z_cm')
        self.declare_parameter('list_yolo_objects_service', '/mcp_server/list_yolo_objects')
        self.declare_parameter('grab_object_service', '/mcp_server/grab_object')
        self.declare_parameter('open_gripper_service', '/mcp_server/open_gripper')
        self.declare_parameter('close_gripper_service', '/mcp_server/close_gripper')
        self.declare_parameter('mcp_service_wait_timeout_sec', 10.0)
        self.declare_parameter('mcp_tool_timeout_sec', 80.0)
        self.declare_parameter('detected_objects_topic', '/object_target_controller/detected_objects')
        self.declare_parameter('yolo_objects_max_age_sec', 5.0)
        self.declare_parameter('llm_resolve_grasp_target', True)
        self.declare_parameter('api_detections_topic', '/mcp_omni_client/api_detections_json')
        self.declare_parameter('target_command_topic', '/economic_grasp_roi/target_class_name')
        self.declare_parameter('api_detection_default_confidence', 0.90)
        self.declare_parameter('api_detection_publish_settle_sec', 0.25)
        self.declare_parameter('api_detection_republish_count', 3)
        self.declare_parameter('api_detection_republish_interval_sec', 0.15)
        self.declare_parameter('api_detection_box_coordinate_space', 'qwen_1000')
        self.declare_parameter('api_detection_box_reference_size', 1000.0)
        self.declare_parameter('grab_api_default_motion_speed', 0.05)

        self.declare_parameter('vision_enabled', True)
        self.declare_parameter('vision_auto_attach_to_turns', False)
        self.declare_parameter('vision_image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('vision_image_max_age_sec', 30.0)
        self.declare_parameter('vision_max_image_width', 640)
        self.declare_parameter('vision_jpeg_quality', 85)
        self.declare_parameter('vision_show_window', True)
        self.declare_parameter('vision_window_name', 'Qwen-Omni Vision Box')
        self.declare_parameter('vision_save_images', True)
        self.declare_parameter('vision_output_dir', '/home/tqq/TQQ_ws/omni_vision_outputs')
        self.declare_parameter('vision_result_hold_sec', 60.0)

    def _read_parameters(self) -> None:
        super()._read_parameters()
        self.omni_api_key_env = str(self.get_parameter('omni_api_key_env').value).strip()
        self.omni_base_url = str(self.get_parameter('omni_base_url').value).strip()
        self.omni_model = str(self.get_parameter('omni_model').value).strip()
        self.omni_text_model = str(self.get_parameter('omni_text_model').value).strip()
        self.omni_realtime_url = str(self.get_parameter('omni_realtime_url').value).strip()
        self.omni_realtime_voice = str(self.get_parameter('omni_realtime_voice').value).strip()
        self.omni_realtime_chunk_bytes = int(self.get_parameter('omni_realtime_chunk_bytes').value)
        self.omni_realtime_chunk_sleep_sec = float(
            self.get_parameter('omni_realtime_chunk_sleep_sec').value
        )
        self.omni_realtime_enable_search = self._as_bool(
            self.get_parameter('omni_realtime_enable_search').value
        )
        self.omni_realtime_connect_retries = int(
            self.get_parameter('omni_realtime_connect_retries').value
        )
        self.omni_realtime_connect_retry_delay_sec = float(
            self.get_parameter('omni_realtime_connect_retry_delay_sec').value
        )
        self.omni_timeout = float(self.get_parameter('omni_timeout').value)
        self.omni_max_tokens = int(self.get_parameter('omni_max_tokens').value)
        self.omni_native_audio_output_enabled = self._as_bool(
            self.get_parameter('omni_native_audio_output_enabled').value
        )
        self.omni_native_audio_fallback_to_local_tts = self._as_bool(
            self.get_parameter('omni_native_audio_fallback_to_local_tts').value
        )
        self.omni_native_audio_sample_rate = int(
            self.get_parameter('omni_native_audio_sample_rate').value
        )
        self.omni_native_audio_silence_bytes = int(
            self.get_parameter('omni_native_audio_silence_bytes').value
        )
        self.omni_native_audio_timeout = float(
            self.get_parameter('omni_native_audio_timeout').value
        )
        self.omni_speech_rate = str(self.get_parameter('omni_speech_rate').value).strip()
        self.omni_speech_volume = str(self.get_parameter('omni_speech_volume').value).strip()
        self.omni_speech_emotion = str(self.get_parameter('omni_speech_emotion').value).strip()
        self.omni_speech_style = str(self.get_parameter('omni_speech_style').value).strip()

        self.go_home_service = str(self.get_parameter('go_home_service').value)
        self.move_x_service = str(self.get_parameter('move_x_service').value)
        self.move_y_service = str(self.get_parameter('move_y_service').value)
        self.move_z_service = str(self.get_parameter('move_z_service').value)
        self.list_yolo_objects_service = str(
            self.get_parameter('list_yolo_objects_service').value
        )
        self.grab_object_service = str(self.get_parameter('grab_object_service').value)
        self.open_gripper_service = str(self.get_parameter('open_gripper_service').value)
        self.close_gripper_service = str(self.get_parameter('close_gripper_service').value)
        self.mcp_service_wait_timeout_sec = float(
            self.get_parameter('mcp_service_wait_timeout_sec').value
        )
        self.mcp_tool_timeout_sec = float(self.get_parameter('mcp_tool_timeout_sec').value)
        self.detected_objects_topic = str(self.get_parameter('detected_objects_topic').value)
        self.yolo_objects_max_age_sec = float(self.get_parameter('yolo_objects_max_age_sec').value)
        self.llm_resolve_grasp_target = self._as_bool(
            self.get_parameter('llm_resolve_grasp_target').value
        )
        self.api_detections_topic = str(self.get_parameter('api_detections_topic').value).strip()
        self.target_command_topic = str(self.get_parameter('target_command_topic').value).strip()
        self.api_detection_default_confidence = float(
            self.get_parameter('api_detection_default_confidence').value
        )
        self.api_detection_publish_settle_sec = max(
            0.0,
            float(self.get_parameter('api_detection_publish_settle_sec').value),
        )
        self.api_detection_republish_count = max(
            1,
            int(self.get_parameter('api_detection_republish_count').value),
        )
        self.api_detection_republish_interval_sec = max(
            0.0,
            float(self.get_parameter('api_detection_republish_interval_sec').value),
        )
        self.api_detection_box_coordinate_space = str(
            self.get_parameter('api_detection_box_coordinate_space').value
        ).strip().lower()
        self.api_detection_box_reference_size = max(
            1.0,
            float(self.get_parameter('api_detection_box_reference_size').value),
        )
        self.grab_api_default_motion_speed = min(
            1.0,
            max(0.0, float(self.get_parameter('grab_api_default_motion_speed').value)),
        )

        self.vision_enabled = self._as_bool(self.get_parameter('vision_enabled').value)
        self.vision_auto_attach_to_turns = self._as_bool(
            self.get_parameter('vision_auto_attach_to_turns').value
        )
        self.vision_image_topic = str(self.get_parameter('vision_image_topic').value).strip()
        self.vision_image_max_age_sec = float(
            self.get_parameter('vision_image_max_age_sec').value
        )
        self.vision_max_image_width = int(self.get_parameter('vision_max_image_width').value)
        self.vision_jpeg_quality = int(self.get_parameter('vision_jpeg_quality').value)
        self.vision_show_window = self._as_bool(self.get_parameter('vision_show_window').value)
        self.vision_window_name = str(self.get_parameter('vision_window_name').value).strip()
        self.vision_save_images = self._as_bool(self.get_parameter('vision_save_images').value)
        self.vision_output_dir = Path(
            str(self.get_parameter('vision_output_dir').value).strip()
        ).expanduser()
        self.vision_result_hold_sec = float(self.get_parameter('vision_result_hold_sec').value)

    def _process_audio_file(self, wav_path: Path):
        self._publish_status(f'omni_audio_input={wav_path}')
        self.transcript_pub.publish(String(data=''))
        response_text = self._run_omni_realtime_tool_turn(wav_path)
        self._publish_response(response_text)
        self._speak_omni_response(response_text)
        return True, response_text

    def _process_text(self, text: str, source: str = 'text'):
        if not text:
            text = self.no_speech_text or '没有收到文本，请再输入一次。'
            self._publish_status('transcript=')
            self.transcript_pub.publish(String(data=''))
            self._publish_response(text)
            self._speak_omni_response(text)
            return True, text

        self._publish_status(f'transcript={text}')
        self.transcript_pub.publish(String(data=text))
        direct_box_call = self._direct_box_tool_call_from_text(text)
        if direct_box_call is not None:
            self._publish_status(
                'direct_visual_box_route='
                f'{direct_box_call["name"]} args={direct_box_call["arguments"]}'
            )
            response_text = self._run_compatible_tool_call(direct_box_call)
            self._publish_response(response_text)
            self._speak_omni_response(response_text)
            return True, response_text
        response_text = self._run_omni_text_tool_turn(text)
        self._publish_response(response_text)
        self._speak_omni_response(response_text)
        return True, response_text

    def _run_omni_realtime_tool_turn(self, wav_path: Path) -> str:
        try:
            import dashscope
            from dashscope.audio.qwen_omni import (
                MultiModality,
                OmniRealtimeCallback,
                OmniRealtimeConversation,
            )
        except ImportError as exc:
            raise RuntimeError(
                'DashScope Realtime SDK is not installed. Install it with: '
                '/usr/bin/python3 -m pip install --user -U "dashscope>=1.23.9" websocket-client'
            ) from exc

        dashscope.api_key = self._api_key_from_env(self.omni_api_key_env)
        pcm_bytes = self._wav_to_realtime_pcm(wav_path)

        done_event = threading.Event()
        lock = threading.Lock()
        text_parts: List[str] = []
        final_text: Dict[str, str] = {'value': ''}
        user_transcript: Dict[str, str] = {'value': ''}
        error_text: Dict[str, str] = {'value': ''}
        function_calls: List[Dict] = []
        node = self

        class RealtimeCallback(OmniRealtimeCallback):
            def on_open(self) -> None:
                node._publish_status('omni_realtime=connected')

            def on_close(self, close_status_code: int, close_msg: str) -> None:
                if not done_event.is_set():
                    error_text['value'] = (
                        f'realtime connection closed: {close_status_code} {close_msg}'
                    )
                    done_event.set()

            def on_event(self, response: dict) -> None:
                event_type = str(response.get('type', ''))
                if event_type == 'error':
                    error_text['value'] = json.dumps(
                        response.get('error', response), ensure_ascii=False
                    )
                    done_event.set()
                    return
                if event_type in ('response.text.delta', 'response.audio_transcript.delta'):
                    delta = str(response.get('delta', ''))
                    if delta:
                        with lock:
                            text_parts.append(delta)
                    return
                if event_type == 'conversation.item.input_audio_transcription.completed':
                    user_transcript['value'] = str(response.get('transcript', '')).strip()
                    if user_transcript['value']:
                        node._publish_status(f'transcript={user_transcript["value"]}')
                        node.transcript_pub.publish(String(data=user_transcript['value']))
                    return
                if event_type == 'response.text.done':
                    final_text['value'] = str(response.get('text', '')).strip()
                    return
                if event_type == 'response.audio_transcript.done':
                    final_text['value'] = str(response.get('transcript', '')).strip()
                    return
                if event_type == 'response.function_call_arguments.done':
                    function_calls.append({
                        'call_id': response.get('call_id') or response.get('item_id') or '',
                        'name': response.get('name') or '',
                        'arguments': response.get('arguments') or '{}',
                    })
                    return
                if event_type == 'response.done':
                    response_obj = response.get('response', {})
                    status = str(response_obj.get('status', ''))
                    if status and status not in ('completed', 'incomplete'):
                        error_text['value'] = json.dumps(
                            response_obj.get('status_details') or response_obj,
                            ensure_ascii=False,
                        )
                    elif not final_text['value']:
                        final_text['value'] = node._extract_realtime_done_text(response)
                    done_event.set()

        conversation = OmniRealtimeConversation(
            model=self.omni_model,
            callback=RealtimeCallback(),
            url=self.omni_realtime_url,
        )

        self._publish_status(f'llm=omni_realtime_tools model={self.omni_model}')
        try:
            self._connect_realtime_with_retry(conversation, 'Qwen-Omni Realtime tool turn')
            update_kwargs = {
                'output_modalities': [MultiModality.TEXT],
                'voice': self.omni_realtime_voice or 'Ethan',
                'enable_turn_detection': False,
                'instructions': self._omni_tool_instructions(),
                'tools': self._omni_tool_schema(),
            }
            if self.omni_realtime_enable_search:
                update_kwargs['enable_search'] = True
                update_kwargs['search_options'] = {'enable_source': True}
            if self.omni_max_tokens > 0:
                update_kwargs['max_tokens'] = self.omni_max_tokens
            conversation.update_session(**update_kwargs)

            self._stream_pcm_to_realtime(conversation, pcm_bytes)
            image_payload = None
            if self.vision_auto_attach_to_turns:
                image_payload = self._latest_camera_jpeg_payload_or_none()
            if image_payload:
                conversation.append_video(image_payload['jpeg_b64'])
            conversation.commit()
            conversation.create_response()
            self._wait_realtime_done(done_event, error_text, 'Qwen-Omni Realtime response')

            all_tool_results = []
            for _ in range(3):
                if not function_calls:
                    answer = self._collected_realtime_text(final_text, text_parts, lock)
                    return self._guard_final_answer(answer, all_tool_results)

                current_calls = list(function_calls)
                tool_results = []
                for function_call in current_calls:
                    function_call = self._rewrite_visual_tool_call_if_needed(
                        user_transcript['value'],
                        function_call,
                    )
                    result_text = self._run_omni_function_call(function_call)
                    tool_results.append(result_text)
                    all_tool_results.append(result_text)
                    if self._tool_result_failed(result_text):
                        break

                post_tool_instruction = self._post_tool_instructions(
                    user_transcript['value'],
                    tool_results,
                )
                for function_call, result_text in zip(current_calls, tool_results):
                    self._send_omni_function_result(
                        conversation,
                        function_call,
                        f'{result_text}\n\n{post_tool_instruction}',
                    )

                failed_results = [text for text in tool_results if self._tool_result_failed(text)]
                if failed_results:
                    return self._format_failed_tool_answer(failed_results)

                self._clear_realtime_buffers(
                    done_event,
                    final_text,
                    text_parts,
                    function_calls,
                    lock,
                )
                conversation.create_response()
                self._wait_realtime_done(
                    done_event,
                    error_text,
                    'Qwen-Omni Realtime final response',
                )

            final_answer = self._collected_realtime_text(final_text, text_parts, lock)
            if function_calls:
                self.get_logger().warn(
                    'Qwen-Omni requested too many nested tools; stopping tool loop.'
                )
            return self._guard_final_answer(
                final_answer or '；'.join(all_tool_results) or '好的。',
                all_tool_results,
            )
        finally:
            try:
                conversation.close()
            except Exception:
                pass

    def _run_omni_text_tool_turn(self, user_text: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
        messages = [
            {'role': 'system', 'content': self._omni_tool_instructions()},
            {'role': 'user', 'content': self._omni_user_content(user_text)},
        ]
        self._publish_status(f'llm=omni_text_tools model={self.omni_text_model}')
        all_tool_results = []
        for _ in range(3):
            response = client.chat.completions.create(
                model=self.omni_text_model,
                messages=messages,
                tools=self._omni_tool_schema(),
                tool_choice='auto',
                modalities=['text'],
                stream=True,
                stream_options={'include_usage': True},
                max_tokens=self.omni_max_tokens,
            )
            answer, tool_calls = self._collect_compatible_stream_response(response)
            if not tool_calls:
                routed_call = self._route_next_text_step_with_llm(
                    client,
                    user_text,
                    answer,
                    all_tool_results,
                )
                if routed_call:
                    if routed_call['name'] == 'final_answer':
                        final_answer = self._final_answer_from_routed_call(routed_call) or answer
                        return self._guard_final_answer(final_answer, all_tool_results)
                    tool_calls = [routed_call]
                    self._publish_status(
                        'omni_tool_router='
                        f'{routed_call["name"]} args={routed_call["arguments"]}'
                    )
                elif answer:
                    return self._guard_final_answer(answer, all_tool_results)
                elif all_tool_results:
                    return '；'.join(all_tool_results)
                else:
                    raise RuntimeError('Qwen-Omni returned an empty response.')

            messages.append({
                'role': 'assistant',
                'content': answer or None,
                'tool_calls': [
                    {
                        'id': tool_call['id'],
                        'type': 'function',
                        'function': {
                            'name': tool_call['name'],
                            'arguments': tool_call['arguments'],
                        },
                    }
                    for tool_call in tool_calls
                ],
            })
            tool_results = []
            for tool_call in tool_calls:
                tool_call = self._rewrite_visual_tool_call_if_needed(user_text, tool_call)
                result_text = self._run_compatible_tool_call(tool_call)
                tool_results.append(result_text)
                all_tool_results.append(result_text)
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'content': result_text,
                })
                if self._tool_result_failed(result_text):
                    break

            failed_results = [text for text in tool_results if self._tool_result_failed(text)]
            if failed_results:
                return self._format_failed_tool_answer(failed_results)

            messages.append({
                'role': 'system',
                'content': self._post_tool_instructions(user_text, tool_results),
            })

        self.get_logger().warn('Qwen-Omni requested too many text tool rounds; stopping tool loop.')
        return self._guard_final_answer('；'.join(all_tool_results) or '好的。', all_tool_results)

    def _connect_realtime_with_retry(self, conversation, label: str) -> None:
        attempts = max(1, self.omni_realtime_connect_retries)
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                conversation.connect()
                if attempt > 1:
                    self.get_logger().info(f'{label} connected on attempt {attempt}.')
                return
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                delay = max(0.0, self.omni_realtime_connect_retry_delay_sec)
                self.get_logger().warn(
                    f'{label} connect failed on attempt {attempt}/{attempts}: {exc}. '
                    f'Retrying in {delay:.1f}s.'
                )
                try:
                    conversation.close()
                except Exception:
                    pass
                if delay > 0.0:
                    time.sleep(delay)
        raise RuntimeError(
            f'{label} websocket connection failed after {attempts} attempts: {last_error}'
        )

    def _guard_final_answer(self, answer: str, tool_results: List[str]) -> str:
        answer = str(answer or '').strip()
        if not answer:
            return ''
        if self._claims_grasp_success(answer) and not any(
            self._is_grab_success_result(result) for result in tool_results
        ):
            if any(self._is_grab_attempt_result(result) for result in tool_results):
                failed_results = [
                    result for result in tool_results
                    if self._is_grab_attempt_result(result) and self._tool_result_failed(result)
                ]
                if failed_results:
                    return self._format_failed_tool_answer(failed_results)
                return '抓取命令已经发送，但还没有确认真正抓取完成。'
            return '我还没有执行抓取工具，不能确认抓取成功。'
        if any(self._is_grab_success_result(result) for result in tool_results):
            return '已发送抓取命令，正在执行抓取。'
        return answer

    def _route_next_text_step_with_llm(
        self,
        client,
        user_text: str,
        draft_answer: str,
        tool_results: List[str],
    ) -> Optional[Dict]:
        if not tool_results and not draft_answer:
            return None
        tool_names = [
            'final_answer',
            'look_camera',
            'go_home',
            'list_api_objects',
            'grab_api_object',
            'open_gripper',
            'close_gripper',
            'move_x_cm',
            'move_y_cm',
            'move_z_cm',
        ]
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是机器人工具路由器。你只判断下一步应该是最终回答还是继续调用一个工具。'
                    '必须基于用户真实意图、已有工具结果和助手草稿，不要做关键词截取。'
                    '普通抓取具体物体时，直接用 grab_api_object；它会自己调用视觉 API 识别目标框。'
                    '如果用户原始请求是普通抓取具体目标，并且还没有 grab_api_object 的结果，'
                    '你必须选择 grab_api_object。'
                    '如果用户只是问“你能看到什么”“画面里有什么”“看见了什么”，必须选择 look_camera，'
                    '不要选择 list_api_objects。'
                    '只有用户问“能抓什么”“有哪些可抓取目标”时，才选择 list_api_objects。'
                    '如果用户只是打开或关闭夹爪，选择 open_gripper 或 close_gripper。'
                    '只能输出 JSON，不要解释。'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'original_user_request': user_text,
                    'assistant_draft_answer': draft_answer,
                    'tool_results_so_far': tool_results,
                    'available_actions': tool_names,
                    'output_schema': {
                        'action': 'final_answer or one available tool name',
                        'arguments': (
                            'object; for grab_api_object use '
                            '{"object_name": raw requested object}'
                        ),
                        'final_answer': 'only when action is final_answer',
                        'reason': 'short Chinese reason',
                    },
                }, ensure_ascii=False),
            },
        ]
        try:
            response = client.chat.completions.create(
                model=self.omni_text_model,
                messages=messages,
                modalities=['text'],
                stream=False,
                max_tokens=400,
                response_format={'type': 'json_object'},
            )
            content = str(response.choices[0].message.content or '').strip()
            data = json.loads(content)
        except Exception as exc:
            self.get_logger().warn(f'LLM tool router failed: {exc}')
            return None

        action = str(data.get('action', '')).strip()
        if action == 'final_answer':
            final_answer = str(data.get('final_answer') or '').strip()
            return {
                'id': f'call_router_{uuid.uuid4().hex}',
                'name': 'final_answer',
                'arguments': json.dumps({'final_answer': final_answer}, ensure_ascii=False),
            }
        valid_tool_names = {name for name in tool_names if name != 'final_answer'}
        if action not in valid_tool_names:
            return None
        arguments = data.get('arguments', {})
        if not isinstance(arguments, dict):
            arguments = {}
        if action == 'grab_api_object' and not str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip():
            arguments = {'object_name': str(user_text or '').strip()}
        return {
            'id': f'call_router_{uuid.uuid4().hex}',
            'name': action,
            'arguments': json.dumps(arguments, ensure_ascii=False),
        }

    def _rewrite_visual_tool_call_if_needed(self, user_text: str, tool_call: Dict) -> Dict:
        name = str(tool_call.get('name') or '').strip()
        if name not in ('list_api_objects', 'list_yolo_objects'):
            return tool_call
        if not self._is_general_vision_question(user_text):
            return tool_call
        rewritten = dict(tool_call)
        rewritten['name'] = 'look_camera'
        rewritten['arguments'] = json.dumps(
            {'question': str(user_text or '').strip() or '请描述当前相机画面。'},
            ensure_ascii=False,
        )
        self._publish_status('visual_tool_rewrite=list_api_objects->look_camera')
        return rewritten

    def _direct_box_tool_call_from_text(self, text: str) -> Optional[Dict]:
        target = self._extract_box_target_from_text(text)
        if not target:
            return None
        return {
            'id': f'call_direct_box_{uuid.uuid4().hex}',
            'name': 'box_api_object',
            'arguments': json.dumps({'object_name': target}, ensure_ascii=False),
        }

    @staticmethod
    def _extract_box_target_from_text(text: str) -> str:
        raw = str(text or '').strip()
        if not raw:
            return ''
        compact = ''.join(raw.split())
        markers = (
            '框选',
            '框出',
            '框住',
            '标出',
            '标注',
            '画框',
            '圈出',
            '圈住',
        )
        marker = next((item for item in markers if item in compact), '')
        if not marker:
            return ''
        target = compact.split(marker, 1)[1]
        cleanup_prefixes = ('一下', '一下子', '这个', '那个', '当前', '画面中', '画面里', '图中', '图里', '的')
        cleanup_suffixes = ('的位置', '的框', '出来', '一下', '吧', '。', '.', '，', ',', '！', '!')
        changed = True
        while changed:
            changed = False
            for prefix in cleanup_prefixes:
                if target.startswith(prefix):
                    target = target[len(prefix):]
                    changed = True
            for suffix in cleanup_suffixes:
                if target.endswith(suffix):
                    target = target[:-len(suffix)]
                    changed = True
        return target.strip()

    @staticmethod
    def _is_general_vision_question(text: str) -> bool:
        compact = ''.join(str(text or '').lower().split())
        if not compact:
            return False
        grasp_markers = (
            '抓',
            '夹',
            '拿',
            '拾取',
            '可抓',
            '能抓',
            '抓取',
            'grasp',
            'grab',
            'pick',
        )
        if any(marker in compact for marker in grasp_markers):
            return False
        vision_markers = (
            '你能看到什么',
            '你能看见什么',
            '能看到什么',
            '能看见什么',
            '看到什么',
            '看见什么',
            '画面里有什么',
            '画面中有什么',
            '图里有什么',
            '图中有什么',
            '相机里有什么',
            '这是什么',
            '有什么',
            'whatdoyousee',
            'whatisintheimage',
        )
        return any(marker in compact for marker in vision_markers)

    @staticmethod
    def _final_answer_from_routed_call(routed_call: Dict) -> str:
        try:
            arguments = json.loads(str(routed_call.get('arguments') or '{}'))
        except json.JSONDecodeError:
            return ''
        return str(arguments.get('final_answer') or '').strip()

    @staticmethod
    def _claims_grasp_success(answer: str) -> bool:
        compact = ''.join(str(answer or '').split())
        if not compact:
            return False
        claim_markers = (
            '抓取成功',
            '成功抓取',
            '成功抓住',
            '成功夹取',
            '成功夹住',
            '成功拿起',
            '成功拾取',
            '已抓取',
            '已抓住',
            '已抓到',
            '已夹取',
            '已夹住',
            '已拿起',
            '已拿到',
            '已拾取',
            '已经抓取',
            '已经抓住',
            '已经抓到',
            '已经夹取',
            '已经夹住',
            '已经拿起',
            '已经拿到',
            '已经拾取',
            '完成抓取',
            '抓取完成',
            '抓取命令已发送',
            '已发送抓取命令',
        )
        return any(marker in compact for marker in claim_markers)

    @staticmethod
    def _is_grab_attempt_result(result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        return (
            '/grab_object' in lowered
            or 'grab_object' in lowered
            or 'grab_api_object' in lowered
        )

    @staticmethod
    def _is_grab_success_result(result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        success_marker = (
            '/grab_object success:' in lowered
            or 'grab_object success:' in lowered
            or 'grab_api_object success:' in lowered
        )
        return (
            success_marker
            and ' failed:' not in lowered
            and ' rejected:' not in lowered
            and ' unavailable' not in lowered
            and ' timed out' not in lowered
        )

    def _omni_user_content(self, user_text: str):
        if self.vision_auto_attach_to_turns:
            return self._omni_user_content_with_camera(user_text)
        return user_text

    def _omni_user_content_with_camera(self, user_text: str):
        content = [{'type': 'text', 'text': user_text}]
        if not self.vision_enabled:
            return user_text
        try:
            image_payload = self._latest_camera_jpeg_payload()
        except Exception as exc:
            self.get_logger().warn(f'Could not attach latest camera image to text turn: {exc}')
            return user_text
        prompt = (
            f'{user_text}\n\n'
            f'注意：随消息发送的图像尺寸是 {image_payload["api_width"]}x'
            f'{image_payload["api_height"]} 像素。'
            f'原始 ROS RGB 图像尺寸是 {image_payload["original_width"]}x'
            f'{image_payload["original_height"]} 像素。'
            '如果要给出像素坐标，必须给出原始 ROS RGB 图像坐标，'
            'x 从左到右，y 从上到下。不要使用归一化坐标。'
        )
        content = [{'type': 'text', 'text': prompt}]
        content.append({'type': 'image_url', 'image_url': {'url': image_payload['data_url']}})
        self._publish_status(
            f'omni_text_image_attached topic={self.vision_image_topic} '
            f'api_size={image_payload["api_width"]}x{image_payload["api_height"]} '
            f'original_size={image_payload["original_width"]}x{image_payload["original_height"]}'
        )
        return content

    def _speak_omni_response(self, text: str) -> None:
        if self.omni_native_audio_output_enabled and self.play_tts:
            generation = self._begin_tts_generation()
            threading.Thread(
                target=self._run_omni_native_audio_thread,
                args=(text, generation),
                daemon=True,
            ).start()
            return
        if self.play_tts:
            self.get_logger().warn(
                'Omni native audio output is disabled; using local TTS. '
                'Set omni_native_audio_output_enabled=true to force model audio.'
            )
            self._speak_text_async(text)

    def _run_omni_native_audio_thread(self, text: str, generation: int) -> None:
        try:
            self._speak_with_omni_realtime_audio(text, generation)
        except Exception as exc:
            if not self._tts_cancelled(generation):
                self.get_logger().error(f'Qwen-Omni native audio failed: {exc}')
                self._publish_status(f'tts_error={exc}')
                if self.omni_native_audio_fallback_to_local_tts:
                    self.get_logger().warn(
                        'Falling back to local TTS because '
                        'omni_native_audio_fallback_to_local_tts=true.'
                    )
                    self._speak_text_async(text)

    def _speak_with_omni_realtime_audio(self, text: str, generation: int) -> None:
        if self._tts_cancelled(generation):
            return
        try:
            import dashscope
            import pyaudio
            from dashscope.audio.qwen_omni import (
                MultiModality,
                OmniRealtimeCallback,
                OmniRealtimeConversation,
            )
        except ImportError as exc:
            raise RuntimeError(
                'DashScope Realtime audio output needs dashscope and pyaudio. Install with: '
                '/usr/bin/python3 -m pip install --user -U dashscope pyaudio'
            ) from exc

        dashscope.api_key = self._api_key_from_env(self.omni_api_key_env)
        done_event = threading.Event()
        error_text: Dict[str, str] = {'value': ''}
        pya = pyaudio.PyAudio()
        node = self

        class AudioCallback(OmniRealtimeCallback):
            def __init__(self) -> None:
                self.out = None

            def on_open(self) -> None:
                self.out = pya.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=node.omni_native_audio_sample_rate,
                    output=True,
                )

            def on_close(self, close_status_code: int, close_msg: str) -> None:
                if not done_event.is_set():
                    error_text['value'] = (
                        f'realtime audio connection closed: {close_status_code} {close_msg}'
                    )
                    done_event.set()

            def on_event(self, response: dict) -> None:
                if node._tts_cancelled(generation):
                    done_event.set()
                    return
                event_type = str(response.get('type', ''))
                if event_type == 'error':
                    error_text['value'] = json.dumps(
                        response.get('error', response), ensure_ascii=False
                    )
                    done_event.set()
                    return
                if event_type == 'response.audio.delta':
                    delta = response.get('delta', '')
                    if delta and self.out is not None:
                        self.out.write(base64.b64decode(delta))
                    return
                if event_type in ('response.audio.done', 'response.done'):
                    done_event.set()

            def close_audio(self) -> None:
                if self.out is not None:
                    try:
                        self.out.stop_stream()
                        self.out.close()
                    finally:
                        self.out = None

        callback = AudioCallback()
        conversation = OmniRealtimeConversation(
            model=self.omni_model,
            callback=callback,
            url=self.omni_realtime_url,
        )
        self._publish_status(
            f'tts=omni_realtime voice={self.omni_realtime_voice} '
            f'rate={self.omni_speech_rate} emotion={self.omni_speech_emotion}'
        )

        try:
            self._connect_realtime_with_retry(conversation, 'Qwen-Omni native audio')
            conversation.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice=self.omni_realtime_voice or 'Ethan',
                enable_turn_detection=False,
                instructions=self._omni_audio_tts_instructions(text),
            )
            prompt_pcm = self._silence_pcm_bytes(max(320, self.omni_native_audio_silence_bytes))
            conversation.append_audio(base64.b64encode(prompt_pcm).decode('ascii'))
            conversation.commit()
            conversation.create_response()
            timeout = max(1.0, self.omni_native_audio_timeout)
            if not done_event.wait(timeout=timeout):
                raise RuntimeError('Qwen-Omni native audio output timed out.')
            if error_text['value'] and not self._tts_cancelled(generation):
                raise RuntimeError(f'Qwen-Omni native audio error: {error_text["value"]}')
        finally:
            try:
                conversation.close()
            except Exception:
                pass
            callback.close_audio()
            pya.terminate()

    def _omni_tool_instructions(self) -> str:
        return (
            '你是 Franka FR3 机械臂的全模态语音控制助手。'
            '你要根据用户真实意图选择工具，而不是从句子里截取关键词。'
            '需要控制机器人或查看相机时，必须使用提供的工具调用，不要把工具调用写成 JSON 文本。'
            '默认用户消息不会携带相机图片。'
            '普通看图、描述颜色、物体关系、场景内容、询问当前画面时，必须调用 look_camera，'
            '只发送文本和当前图片并接收 API 原始文本回复；不要进行本地图像处理，不要画框，不要保存图片。'
            '机器人抓取和可抓取目标查询不使用本地 YOLO，所有图像识别都走视觉 API。'
            '当用户是在询问当前能检测到什么、当前有哪些可抓取目标、机器人现在能抓哪些东西时，'
            '必须调用 list_api_objects，并把视觉 API 返回的列表总结给用户；'
            '这类问题是“查询能力/列表”，不是抓取动作，不能把疑问代词当成物体名称。'
            '当用户明确要求抓取、拿起、夹取、拾取某个具体目标时，默认调用 grab_api_object。'
            '如果用户在抓取命令里指定速度，例如“用0.2的速度抓橘子”，'
            'grab_api_object 要填写 motion_speed=0.2；这个速度只对本次抓取生效。'
            '如果用户没有指定速度，系统会自动使用默认抓取速度，不要主动询问速度。'
            '当用户只要求打开、松开、关闭或合上夹爪时，必须调用 open_gripper 或 close_gripper，'
            '不要调用 grab_api_object。'
            '只调用 list_api_objects 不能算抓取成功；只有 grab_api_object 工具返回 success，'
            '才能说已经发送抓取命令。'
            'grab_api_object 的 object_name 填用户原话里的目标描述即可；'
            '系统会在工具执行时把当前相机图发送给视觉 API，让 API 返回目标检测框。'
            '抓取工具会在本地把 API 检测框作为 RGB-D ROI，分割目标点云，'
            '用 EconomicGrasp 生成抓取位姿，再通过 TF 到基座标、MoveIt IK 执行并关闭夹爪；'
            '你不能自己编造像素坐标或三维坐标来抓。'
            '方向约定：右/向右是 x 正方向，左/向左是 x 负方向；'
            '前/向前是 y 正方向，后/向后是 y 负方向；'
            '上/向上是 z 正方向，下/向下是 z 负方向。'
            '用户说厘米时，centimeters 使用厘米数；负方向用负数。'
            '如果工具执行失败、拒绝、超时或不可用，只能告诉用户本次命令没有执行成功，'
            '不能自动拆分成多步继续执行。'
            '如果只是聊天或普通提问，不需要调用工具，直接简洁回答。'
        )

    @staticmethod
    def _post_tool_instructions(user_text: str, tool_results: List[str]) -> str:
        original_request = str(user_text or '').strip() or '当前这次用户请求'
        return (
            '刚才的工具调用已经返回。请回看原始用户请求，继续用意图而不是关键词判断下一步。'
            f'原始用户请求：{original_request}\n'
            f'刚才工具结果：{"；".join(tool_results)}\n'
            '如果原始请求只是询问当前能看到什么、画面里有什么或看见了什么，'
            'look_camera 已经足够，直接用 API 文本结果回答。'
            '如果原始请求只是询问当前能抓什么或有哪些可抓取目标，'
            'list_api_objects 已经足够，直接总结可抓取目标。'
            '如果原始请求是普通抓取、拿起、夹取、拾取某个具体目标，'
            '必须调用 grab_api_object 执行抓取。'
            '如果原始请求只是打开/关闭/松开/合上夹爪，'
            '只总结 open_gripper 或 close_gripper 的执行结果。'
            '没有 grab_api_object success 时，不要说已经抓取成功；'
            '不要输出 JSON 或 XML。'
        )

    def _omni_audio_tts_instructions(self, text: str) -> str:
        return (
            '你是机器人语音播报器。请只把下面这段中文回复用语音自然念出来，'
            '不要添加额外解释，不要提到系统提示，不要输出 JSON。'
            f'音色={self.omni_realtime_voice or "Ethan"}；'
            f'语速={self.omni_speech_rate or "normal"}；'
            f'音量={self.omni_speech_volume or "normal"}；'
            f'情绪={self.omni_speech_emotion or "natural"}；'
            f'风格={self.omni_speech_style or "自然清晰"}。\n'
            f'需要朗读的内容：\n{text}'
        )

    def _omni_tool_schema(self) -> List[Dict]:
        centimeter_property = {
            'type': 'object',
            'properties': {
                'centimeters': {
                    'type': 'number',
                    'description': (
                        'Distance in centimeters. Positive follows the named base-frame axis; '
                        'negative moves in the opposite direction.'
                    ),
                },
            },
            'required': ['centimeters'],
            'additionalProperties': False,
        }
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'look_camera',
                    'description': (
                        'Use the latest RealSense camera image to answer questions about '
                        'what is visible, object identity, color, relative position, and '
                        'visual distance estimates.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'The user question to answer using the current camera image.',
                            },
                        },
                        'required': ['question'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'go_home',
                    'description': 'Move the robot arm back to the saved home joint pose.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_api_objects',
                    'description': (
                        'Use the latest camera image and vision API to return graspable object '
                        'candidates with boxes. Use this only when the user asks what objects '
                        'are available for grasping or what the robot can grab. Do not use it '
                        'for general scene description questions such as 你能看到什么.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'Optional user question about visible/graspable objects.',
                            },
                        },
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'grab_api_object',
                    'description': (
                        'Grasp one object by using the vision API to detect its bounding box, '
                        'then using that box as an RGB-D ROI for EconomicGrasp pose generation, '
                        'TF to the robot base frame, and MoveIt IK. '
                        'Do not provide pixel coordinates; only provide the object name.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_name': {
                                'type': 'string',
                                'description': (
                                    'Raw user target description for the object to grab, for example '
                                    '橘子, 甜甜圈, 中间那个, orange, donut. The client will send the '
                                    'current camera image to the vision API and ask for this target box.'
                                ),
                            },
                            'motion_speed': {
                                'type': 'number',
                                'description': (
                                    'Optional one-shot motion speed scaling for this grasp only, '
                                    'from 0.0 to 1.0. Use this only when the user explicitly says '
                                    'a speed such as 用0.2的速度抓橘子. Omit it to use the default speed.'
                                ),
                            },
                        },
                        'required': ['object_name'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'box_api_object',
                    'description': (
                        'Draw boxes for a requested object in the Qwen-Omni Vision Box. '
                        'Use this when the user asks to frame, box, mark, annotate, or '
                        'circle an object, but does not ask the robot to grasp it.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'object_name': {
                                'type': 'string',
                                'description': 'Target object to draw a box around, for example 苹果 or apple.',
                            },
                        },
                        'required': ['object_name'],
                        'additionalProperties': False,
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'open_gripper',
                    'description': 'Open the Franka gripper to the configured open width.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'close_gripper',
                    'description': 'Close the Franka gripper with the configured grasp force.',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'move_x_cm',
                    'description': 'Move the end effector along base-frame X. Right is positive; left is negative.',
                    'parameters': centimeter_property,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'move_y_cm',
                    'description': 'Move the end effector along base-frame Y. Forward is positive; backward is negative.',
                    'parameters': centimeter_property,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'move_z_cm',
                    'description': 'Move the end effector along base-frame Z. Up is positive; down is negative.',
                    'parameters': centimeter_property,
                },
            },
        ]

    def _run_compatible_tool_call(self, tool_call: Dict) -> str:
        name = str(tool_call.get('name') or '')
        raw_arguments = tool_call.get('arguments') or '{}'
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        return self._run_omni_named_tool(name, arguments)

    def _run_omni_function_call(self, function_call: Dict) -> str:
        name = str(function_call.get('name') or '').strip()
        raw_arguments = function_call.get('arguments') or '{}'
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(str(raw_arguments))
            except json.JSONDecodeError:
                arguments = {}
        return self._run_omni_named_tool(name, arguments)

    def _run_omni_named_tool(self, name: str, arguments: Dict) -> str:
        if not isinstance(arguments, dict):
            arguments = {}
        self._publish_status(f'omni_tool_call={name} args={json.dumps(arguments, ensure_ascii=False)}')
        if name == 'look_camera':
            return self._call_look_camera(arguments)
        if name == 'go_home':
            return self._call_trigger(self.go_home_client, self.go_home_service)
        if name in ('list_api_objects', 'list_yolo_objects'):
            return self._call_list_api_objects(arguments)
        if name == 'box_api_object':
            return self._call_box_api_object(arguments)
        if name in ('grab_api_object', 'grab_yolo_object'):
            return self._call_grab_api_object(arguments)
        if name == 'open_gripper':
            return self._call_trigger(self.open_gripper_client, self.open_gripper_service)
        if name == 'close_gripper':
            return self._call_trigger(self.close_gripper_client, self.close_gripper_service)
        if name == 'move_x_cm':
            return self._call_move_axis(self.move_x_client, self.move_x_service, arguments)
        if name == 'move_y_cm':
            return self._call_move_axis(self.move_y_client, self.move_y_service, arguments)
        if name == 'move_z_cm':
            return self._call_move_axis(self.move_z_client, self.move_z_service, arguments)
        return f'unsupported omni tool: {name or "empty"}'

    def _call_look_camera(self, arguments: Dict) -> str:
        if not self.vision_enabled:
            return 'look_camera failed: vision is disabled'
        question = str(arguments.get('question') or '').strip() or '请描述当前相机画面。'
        try:
            answer = self._ask_vision_model(question)
        except Exception as exc:
            return f'look_camera failed: {exc}'
        return f'look_camera success: {answer}'

    def _ask_vision_model(self, question: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        image_payload = self._latest_camera_jpeg_payload()
        prompt = (
            f'{question}\n\n'
            f'图像发送尺寸是 {image_payload["api_width"]}x{image_payload["api_height"]} 像素。'
            f'原始 ROS RGB 图像尺寸是 {image_payload["original_width"]}x'
            f'{image_payload["original_height"]} 像素。'
            '如果需要给出像素坐标，必须使用原始 ROS RGB 图像坐标，'
            'x 从左到右，y 从上到下，不要使用归一化坐标。'
        )
        self._publish_status(
            f'omni_vision_call model={self.omni_text_model} topic={self.vision_image_topic} '
            f'api_size={image_payload["api_width"]}x{image_payload["api_height"]} '
            f'original_size={image_payload["original_width"]}x{image_payload["original_height"]}'
        )
        client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
        response = client.chat.completions.create(
            model=self.omni_text_model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '你是机器人相机视觉助手。只根据用户问题和当前图像回答，'
                        '不要调用机器人控制工具，不要编造抓取成功。'
                    ),
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {'url': image_payload['data_url']},
                        },
                    ],
                },
            ],
            modalities=['text'],
            stream=False,
            max_tokens=self.omni_max_tokens,
        )
        return str(response.choices[0].message.content or '').strip() or '没有得到视觉回复。'

    def _call_trigger(self, client, service_name: str) -> str:
        if not client.wait_for_service(timeout_sec=self.mcp_service_wait_timeout_sec):
            return f'{service_name} unavailable'
        future = client.call_async(Trigger.Request())
        response = self._wait_for_service_future(future, service_name)
        if response is None:
            return f'{service_name} timed out'
        status = 'success' if response.success else 'failed'
        return f'{service_name} {status}: {response.message}'

    def _call_move_axis(self, client, service_name: str, arguments: Dict) -> str:
        if not client.wait_for_service(timeout_sec=self.mcp_service_wait_timeout_sec):
            return f'{service_name} unavailable'
        request = MoveAxis.Request()
        request.centimeters = float(arguments.get('centimeters', 0.0))
        future = client.call_async(request)
        response = self._wait_for_service_future(future, service_name)
        if response is None:
            return f'{service_name} timed out'
        status = 'success' if response.success else 'failed'
        return f'{service_name} {status}: {response.message}'

    def _call_list_api_objects(self, arguments: Dict) -> str:
        if not self.vision_enabled:
            return 'list_api_objects failed: vision is disabled'
        self._publish_api_target_command('', None)
        question = str(arguments.get('question') or '').strip()
        if not question:
            question = '请列出当前画面里清晰可见、适合机械臂抓取的物体。'
        try:
            detections = self._detect_objects_with_vision_api(question, target_name='')
        except Exception as exc:
            return f'list_api_objects failed: {exc}'
        if not detections:
            return 'list_api_objects success: 当前视觉 API 没有返回可抓取目标。'
        self._save_and_show_api_detection_result(
            detections,
            answer={'objects': detections},
            mode='list_api_objects',
            status='Qwen-Omni API vision result',
        )
        parts = []
        for item in detections:
            name = str(item.get('class_name', '')).strip() or 'object'
            confidence = float(item.get('confidence', 0.0))
            bbox = item.get('bbox_xyxy', [])
            if len(bbox) == 4:
                parts.append(
                    f'{name} confidence={confidence:.2f} '
                    f'bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]'
                )
            else:
                parts.append(f'{name} confidence={confidence:.2f}')
        return 'list_api_objects success: vision API detected: ' + ', '.join(parts)

    def _call_box_api_object(self, arguments: Dict) -> str:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        if not requested_name:
            return 'box_api_object failed: object name is empty'
        if not self.vision_enabled:
            return 'box_api_object failed: vision is disabled'

        try:
            detections = self._detect_objects_with_vision_api(
                f'请框出画面中的“{requested_name}”。如果有多个符合的目标，请分别返回多个框。',
                target_name=requested_name,
                max_results=0,
            )
        except Exception as exc:
            return f'box_api_object failed: {exc}'
        if not detections:
            return f'box_api_object failed: 视觉 API 没有找到“{requested_name}”。'

        output_path = self._save_and_show_api_detection_result(
            detections,
            answer={'objects': detections},
            mode='box_api_object',
            status='Qwen-Omni box result',
        )
        parts = []
        for item in detections:
            name = str(item.get('class_name', '')).strip() or requested_name
            bbox = item.get('bbox_xyxy', [])
            if len(bbox) == 4:
                parts.append(
                    f'{name} bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]'
                )
            else:
                parts.append(name)
        return (
            f'box_api_object success: 已在 Vision Box 框出 {len(detections)} 个'
            f'“{requested_name}”：' + '，'.join(parts)
            + f'；saved={output_path or "disabled"}'
        )

    def _call_grab_api_object(self, arguments: Dict) -> str:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        if not requested_name:
            return 'grab_api_object failed: object name is empty'
        if not self.vision_enabled:
            return 'grab_api_object failed: vision is disabled'
        if self.target_command_pub.get_subscription_count() < 1:
            return (
                'grab_api_object failed: no subscriber on '
                f'{self.target_command_topic}; start roi_economic_grasp_controller first'
            )
        if self.api_detections_pub.get_subscription_count() < 1:
            return (
                'grab_api_object failed: no subscriber on '
                f'{self.api_detections_topic}; start roi_economic_grasp_controller first'
            )

        try:
            detections = self._detect_objects_with_vision_api(
                f'请只定位最符合“{requested_name}”的一个目标。',
                target_name=requested_name,
            )
        except Exception as exc:
            return f'grab_api_object failed: {exc}'
        if not detections:
            return f'grab_api_object failed: 视觉 API 没有找到“{requested_name}”。'

        detection = detections[0]
        target_name = str(detection.get('class_name', '')).strip() or requested_name
        speed = self._optional_motion_speed(arguments)
        if speed is None:
            speed = self.grab_api_default_motion_speed
        self._publish_api_target_command('', None)
        if self.api_detection_publish_settle_sec > 0.0:
            time.sleep(self.api_detection_publish_settle_sec)
        self._publish_api_target_command(target_name, speed)
        if self.api_detection_publish_settle_sec > 0.0:
            time.sleep(self.api_detection_publish_settle_sec)
        payload = self._api_detection_payload([detection])
        self._publish_api_detection_payload(payload)
        output_path = self._save_and_show_api_detection_result(
            [detection],
            answer=payload,
            mode='grab_api_object',
            status=f'Qwen-Omni target: {self._ascii_for_cv_text(target_name) or "object"}',
        )

        bbox = detection.get('bbox_xyxy', [0.0, 0.0, 0.0, 0.0])
        message = (
            f'grab_api_object command sent: requested="{requested_name}", '
            f'target="{target_name}", motion_speed={speed if speed is not None else "default"}, '
            f'bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]. '
            f'published detections to {self.api_detections_topic}; '
            f'saved={output_path or "disabled"}. '
            'roi_economic_grasp_controller will use the API bbox as ROI, segment RGB-D points, '
            'run EconomicGrasp, then use MoveIt IK and execute the motion.'
        )
        self._publish_status(message)
        return f'grab_api_object success: {message}'

    def _detect_objects_with_vision_api(
        self,
        question: str,
        target_name: str = '',
        max_results: int = 1,
    ) -> List[Dict]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        image_payload = self._latest_camera_jpeg_payload()
        self._latest_api_detection_image_header = {
            'stamp': {
                'sec': int(image_payload.get('stamp_sec', 0)),
                'nanosec': int(image_payload.get('stamp_nanosec', 0)),
            },
            'frame_id': str(image_payload.get('frame_id', '')),
        }
        width = int(image_payload['original_width'])
        height = int(image_payload['original_height'])
        if target_name:
            task = (
                f'目标物体是“{target_name}”。只返回最适合抓取的一个目标框；'
                '如果没看到这个目标，objects 返回空数组。'
            )
        else:
            task = (
                '返回当前画面中清晰可见、适合机械臂抓取的物体列表。'
                '同类多个物体要分别给多个框，不要用一个大框包住多个物体。'
            )
        prompt = (
            f'{question}\n'
            f'{task}\n'
            f'原始图像尺寸是 width={width}, height={height}。'
            '如果返回 bbox_xyxy，请使用 0 到 1000 的视觉坐标系：'
            'xmin,ymin,xmax,ymax 都是相对于 1000x1000 图像的整数。'
            '请估计每个物体的轴对齐矩形框，尽量贴合物体外轮廓。'
            'class_name 必须是英文小写名词，例如 orange/apple/cup。'
            '只输出严格 JSON，不要 Markdown，不要解释，不要思考过程。'
            'JSON 格式：'
            '{"objects":[{"class_name":"english_label","label_zh":"中文名",'
            '"confidence":0.0到1.0,"bbox_xyxy":[xmin,ymin,xmax,ymax]}]}'
        )
        self._publish_status(
            f'omni_api_detection_call model={self.omni_text_model} target={target_name or "list"} '
            f'topic={self.vision_image_topic} size={width}x{height}'
        )
        client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
        response = client.chat.completions.create(
            model=self.omni_text_model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        '你是机器人抓取视觉检测器。你的唯一任务是根据图像返回物体检测框 JSON。'
                        '不要生成图片，不要描述过程，不要输出 JSON 之外的任何文字。'
                    ),
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_payload['data_url']}},
                    ],
                },
            ],
            modalities=['text'],
            stream=False,
            max_tokens=min(max(300, self.omni_max_tokens), 1200),
            response_format={'type': 'json_object'},
        )
        content = str(response.choices[0].message.content or '').strip()
        data = self._load_json_object(content)
        detections = self._detections_from_api_json(data, width, height)
        self._last_api_detection_raw_json = data
        if target_name and max_results > 0 and detections:
            return detections[:max_results]
        return detections

    def _load_json_object(self, content: str) -> Dict:
        text = str(content or '').strip()
        if not text:
            raise RuntimeError('vision API returned empty JSON')
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start < 0 or end <= start:
                raise RuntimeError(f'vision API did not return JSON: {text[:160]}')
            data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise RuntimeError('vision API JSON root is not an object')
        return data

    def _detections_from_api_json(self, data: Dict, width: int, height: int) -> List[Dict]:
        objects = data.get('objects', [])
        if isinstance(objects, dict):
            objects = [objects]
        if not isinstance(objects, list):
            objects = []

        detections = []
        for index, item in enumerate(objects):
            if not isinstance(item, dict):
                continue
            class_name = str(
                item.get('class_name')
                or item.get('label_en')
                or item.get('name_en')
                or item.get('label')
                or item.get('name')
                or 'object'
            ).strip().lower()
            if not class_name:
                class_name = 'object'
            bbox, normalized_bbox, raw_bbox, bbox_scale = self._bbox_from_api_item(item, width, height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            confidence = self._api_confidence(item)
            detections.append({
                'class_id': index,
                'class_name': class_name,
                'label_zh': str(item.get('label_zh') or item.get('name_zh') or '').strip(),
                'confidence': confidence,
                'bbox_xyxy': [x1, y1, x2, y2],
                'bbox_normalized': normalized_bbox,
                'bbox_raw': raw_bbox,
                'bbox_scale': bbox_scale,
                'center_xy': [center_x, center_y],
                'center_xy_int': [int(round(center_x)), int(round(center_y))],
            })
        return detections

    def _bbox_from_api_item(
        self,
        item: Dict,
        width: int,
        height: int,
    ) -> Tuple[Optional[List[float]], List[float], List[float], str]:
        raw = item.get('bbox_xyxy') or item.get('xyxy')
        scale = self.api_detection_box_coordinate_space or 'qwen_1000'
        if raw is None:
            raw = item.get('bbox_percent') or item.get('box_percent')
            scale = 'percent'
        if raw is None:
            raw = item.get('bbox_normalized') or item.get('box_normalized')
            scale = 'normalized'
        if raw is None:
            raw = item.get('box') or item.get('bbox')
            scale = self._infer_bbox_scale(raw)
        if isinstance(raw, dict):
            raw = [
                raw.get('xmin', raw.get('x1')),
                raw.get('ymin', raw.get('y1')),
                raw.get('xmax', raw.get('x2')),
                raw.get('ymax', raw.get('y2')),
            ]
        if not isinstance(raw, list) or len(raw) != 4:
            return None, [], [], scale
        try:
            x1, y1, x2, y2 = [float(value) for value in raw]
        except (TypeError, ValueError):
            return None, [], [], scale
        raw_bbox = [x1, y1, x2, y2]

        if scale in ('qwen_1000', '1000', 'vlm_1000'):
            ref = self.api_detection_box_reference_size
            nx1 = x1 / ref
            ny1 = y1 / ref
            nx2 = x2 / ref
            ny2 = y2 / ref
            x1 = nx1 * width
            x2 = nx2 * width
            y1 = ny1 * height
            y2 = ny2 * height
        elif scale == 'pixel':
            nx1 = x1 / float(width)
            nx2 = x2 / float(width)
            ny1 = y1 / float(height)
            ny2 = y2 / float(height)
        elif scale == 'normalized':
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
        elif scale == 'percent':
            nx1 = x1 / 100.0
            nx2 = x2 / 100.0
            ny1 = y1 / 100.0
            ny2 = y2 / 100.0
            x1 = nx1 * width
            x2 = nx2 * width
            y1 = ny1 * height
            y2 = ny2 * height
        else:
            inferred = self._infer_bbox_scale(raw)
            return self._bbox_from_api_values(raw_bbox, inferred, width, height)

        normalized = [
            max(0.0, min(1.0, min(nx1, nx2))),
            max(0.0, min(1.0, min(ny1, ny2))),
            max(0.0, min(1.0, max(nx1, nx2))),
            max(0.0, min(1.0, max(ny1, ny2))),
        ]
        left = max(0.0, min(float(width - 1), min(x1, x2)))
        right = max(0.0, min(float(width - 1), max(x1, x2)))
        top = max(0.0, min(float(height - 1), min(y1, y2)))
        bottom = max(0.0, min(float(height - 1), max(y1, y2)))
        if right - left < 2.0 or bottom - top < 2.0:
            return None, normalized, raw_bbox, scale
        return [left, top, right, bottom], normalized, raw_bbox, scale

    def _bbox_from_api_values(
        self,
        raw_bbox: List[float],
        scale: str,
        width: int,
        height: int,
    ) -> Tuple[Optional[List[float]], List[float], List[float], str]:
        item = {'bbox_xyxy': raw_bbox}
        previous = self.api_detection_box_coordinate_space
        try:
            self.api_detection_box_coordinate_space = scale
            return self._bbox_from_api_item(item, width, height)
        finally:
            self.api_detection_box_coordinate_space = previous

    @staticmethod
    def _infer_bbox_scale(raw) -> str:
        values = []
        if isinstance(raw, dict):
            raw_values = [
                raw.get('xmin', raw.get('x1')),
                raw.get('ymin', raw.get('y1')),
                raw.get('xmax', raw.get('x2')),
                raw.get('ymax', raw.get('y2')),
            ]
        else:
            raw_values = raw if isinstance(raw, list) else []
        for value in raw_values:
            try:
                values.append(abs(float(value)))
            except (TypeError, ValueError):
                pass
        if len(values) == 4 and max(values) <= 1.5:
            return 'normalized'
        return 'pixel'

    def _api_confidence(self, item: Dict) -> float:
        raw = item.get('confidence', item.get('score', self.api_detection_default_confidence))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = self.api_detection_default_confidence
        return min(1.0, max(0.0, value))

    def _api_detection_payload(self, detections: List[Dict]) -> Dict:
        now = self.get_clock().now().to_msg()
        header = self._latest_api_detection_image_header
        if not isinstance(header, dict):
            header = {
                'stamp': {
                    'sec': int(now.sec),
                    'nanosec': int(now.nanosec),
                },
                'frame_id': self.vision_image_topic,
            }
        return {
            'source': 'mcp_omni_api',
            'header': header,
            'detections': detections,
        }

    def _publish_api_detection_payload(self, payload: Dict) -> None:
        text = json.dumps(payload, ensure_ascii=True)
        count = max(1, self.api_detection_republish_count)
        for index in range(count):
            self.api_detections_pub.publish(String(data=text))
            if index + 1 < count and self.api_detection_republish_interval_sec > 0.0:
                time.sleep(self.api_detection_republish_interval_sec)

    def _publish_api_target_command(
        self,
        target_name: str,
        motion_speed: Optional[float],
    ) -> None:
        target_name = str(target_name or '').strip()
        if not target_name:
            self.target_command_pub.publish(String(data=''))
            return
        if motion_speed is None:
            payload = target_name
        else:
            payload = json.dumps(
                {
                    'name': target_name,
                    'motion_speed': motion_speed,
                },
                ensure_ascii=False,
            )
        self.target_command_pub.publish(String(data=payload))

    def _call_grab_yolo_object(self, client, service_name: str, arguments: Dict) -> str:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        if self.llm_resolve_grasp_target:
            ok, refresh_message = self._ensure_fresh_yolo_objects_for_grasp()
            if not ok:
                return f'{service_name} failed: {refresh_message}'
            resolved_name, reason = self._resolve_grasp_target_with_llm(requested_name)
            if not resolved_name:
                return f'{service_name} failed: {reason}'
            if resolved_name != requested_name:
                self._publish_status(
                    f'grasp_target_resolved requested={requested_name} yolo_class={resolved_name}'
                )
            arguments = dict(arguments)
            arguments['object_name'] = resolved_name
        return self._call_object_name(client, service_name, arguments)

    def _ensure_fresh_yolo_objects_for_grasp(self) -> Tuple[bool, str]:
        if self._latest_yolo_object_items():
            return True, 'using cached YOLO object list'

        self._publish_status(
            'grab_yolo_object needs a fresh YOLO list; calling list_yolo_objects first'
        )
        since_time = time.monotonic()
        if not self.list_yolo_objects_client.wait_for_service(
            timeout_sec=self.mcp_service_wait_timeout_sec,
        ):
            return False, (
                f'{self.list_yolo_objects_service} unavailable; cannot refresh YOLO before grab'
            )

        future = self.list_yolo_objects_client.call_async(Trigger.Request())
        response = self._wait_for_service_future(future, self.list_yolo_objects_service)
        if response is None:
            return False, f'{self.list_yolo_objects_service} timed out before grab'
        if not response.success:
            return False, (
                f'{self.list_yolo_objects_service} failed before grasp: {response.message}'
            )

        fresh_seen, items = self._wait_for_fresh_yolo_object_items(
            since_time,
            min(2.0, max(0.5, self.mcp_service_wait_timeout_sec)),
        )
        if not fresh_seen:
            return False, (
                f'{self.list_yolo_objects_service} succeeded but no fresh '
                f'{self.detected_objects_topic} message was received before grab: '
                f'{response.message}'
            )
        if not items:
            return False, f'当前 YOLO 没有检测到可抓取目标：{response.message}'
        return True, response.message

    def _call_object_name(self, client, service_name: str, arguments: Dict) -> str:
        if not client.wait_for_service(timeout_sec=self.mcp_service_wait_timeout_sec):
            return f'{service_name} unavailable'
        request = ObjectName.Request()
        name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        speed = self._optional_motion_speed(arguments)
        if speed is None:
            request.name = name
        else:
            request.name = json.dumps(
                {
                    'name': name,
                    'motion_speed': speed,
                },
                ensure_ascii=False,
            )
        future = client.call_async(request)
        response = self._wait_for_service_future(future, service_name)
        if response is None:
            return f'{service_name} timed out'
        status = 'success' if response.success else 'failed'
        return f'{service_name} {status}: {response.message}'

    @staticmethod
    def _optional_motion_speed(arguments: Dict) -> Optional[float]:
        raw = (
            arguments.get('motion_speed')
            if 'motion_speed' in arguments
            else arguments.get('speed')
        )
        if raw is None or raw == '':
            return None
        try:
            speed = float(raw)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, speed))

    def _yolo_objects_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {'objects': [], 'raw': msg.data}
        with self._yolo_objects_lock:
            self._latest_yolo_objects_payload = payload
            self._latest_yolo_objects_time = time.monotonic()

    def _latest_yolo_object_items(self) -> List[Dict]:
        with self._yolo_objects_lock:
            payload = self._latest_yolo_objects_payload
            stamp = self._latest_yolo_objects_time
        if not isinstance(payload, dict):
            return []
        age = time.monotonic() - stamp
        if self.yolo_objects_max_age_sec > 0.0 and age > self.yolo_objects_max_age_sec:
            self.get_logger().warn(
                f'YOLO object list is stale: age={age:.2f}s on {self.detected_objects_topic}'
            )
            return []
        objects = payload.get('objects', [])
        return [item for item in objects if isinstance(item, dict)] if isinstance(objects, list) else []

    def _wait_for_fresh_yolo_object_items(
        self,
        since_time: float,
        timeout_sec: float,
    ) -> Tuple[bool, List[Dict]]:
        deadline = time.monotonic() + max(0.1, timeout_sec)
        while time.monotonic() < deadline:
            with self._yolo_objects_lock:
                payload = self._latest_yolo_objects_payload
                stamp = self._latest_yolo_objects_time
            if isinstance(payload, dict) and stamp >= since_time:
                objects = payload.get('objects', [])
                items = (
                    [item for item in objects if isinstance(item, dict)]
                    if isinstance(objects, list)
                    else []
                )
                return True, items
            time.sleep(0.02)
        return False, []

    def _resolve_grasp_target_with_llm(self, requested_name: str) -> Tuple[str, str]:
        if not requested_name:
            return '', 'empty grasp target'

        objects = self._latest_yolo_object_items()
        if not objects:
            return '', (
                f'当前没有收到可用的 YOLO 目标列表，请先确认 {self.detected_objects_topic} 正在发布。'
            )

        candidates = []
        for item in objects:
            class_name = str(item.get('class_name', '')).strip()
            if not class_name:
                continue
            candidates.append({
                'class_name': class_name,
                'class_id': item.get('class_id', -1),
                'count': item.get('count', 1),
                'confidence': item.get('best_confidence', item.get('confidence', 0.0)),
            })
        if not candidates:
            return '', '当前 YOLO 列表为空，无法选择抓取目标。'

        direct = self._direct_yolo_name_match(requested_name, candidates)
        if direct:
            return direct, 'direct match'

        try:
            resolved_name, reason = self._ask_llm_to_resolve_yolo_class(requested_name, candidates)
        except Exception as exc:
            return '', f'LLM 目标对齐失败：{exc}'

        if not resolved_name:
            available = ', '.join(item['class_name'] for item in candidates)
            return '', f'无法把“{requested_name}”匹配到当前 YOLO 目标。当前可抓取目标：{available}'

        valid_names = {item['class_name'] for item in candidates}
        if resolved_name not in valid_names:
            available = ', '.join(sorted(valid_names))
            return '', (
                f'LLM 返回了不在当前 YOLO 列表中的目标“{resolved_name}”。'
                f'当前可选：{available}。原因：{reason}'
            )
        return resolved_name, reason

    @staticmethod
    def _direct_yolo_name_match(requested_name: str, candidates: List[Dict]) -> str:
        request_lower = requested_name.strip().lower()
        for item in candidates:
            class_name = str(item.get('class_name', '')).strip()
            if class_name.lower() == request_lower:
                return class_name
        return ''

    def _ask_llm_to_resolve_yolo_class(
        self,
        requested_name: str,
        candidates: List[Dict],
    ) -> Tuple[str, str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed for this Python. Install it with: '
                '/usr/bin/python3 -m pip install --user -U openai'
            ) from exc

        client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
        messages = [
            {
                'role': 'system',
                'content': (
                    '你负责把用户想抓取的自然语言目标，匹配到当前 YOLO 检测列表中的一个精确 class_name。'
                    '只能从候选 class_name 中选择，不能创造新类别。'
                    '如果没有可靠匹配，返回 selected_class_name 为空字符串。'
                    '只输出 JSON，不要解释。'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'requested_object': requested_name,
                    'yolo_candidates': candidates,
                    'output_schema': {
                        'selected_class_name': 'one candidate class_name or empty string',
                        'confidence': 'number from 0 to 1',
                        'reason': 'short Chinese reason',
                    },
                }, ensure_ascii=False),
            },
        ]
        self._publish_status(
            f'llm=resolve_yolo_target model={self.omni_text_model} requested={requested_name}'
        )
        response = client.chat.completions.create(
            model=self.omni_text_model,
            messages=messages,
            modalities=['text'],
            stream=False,
            max_tokens=300,
            response_format={'type': 'json_object'},
        )
        content = str(response.choices[0].message.content or '').strip()
        data = json.loads(content)
        selected = str(data.get('selected_class_name', '')).strip()
        reason = str(data.get('reason', '')).strip()
        confidence = float(data.get('confidence', 0.0) or 0.0)
        if confidence < 0.45:
            return '', reason or f'匹配置信度过低：{confidence:.2f}'
        return selected, reason or f'confidence={confidence:.2f}'

    def _wait_for_service_future(self, future, label: str):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=max(0.1, self.mcp_tool_timeout_sec)):
            self.get_logger().error(f'Timed out waiting for {label}.')
            return None
        try:
            return future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed waiting for {label}: {exc}')
            return None

    def _vision_image_callback(self, msg: RosImage) -> None:
        with self._vision_image_lock:
            self._latest_vision_image = msg
            self._latest_vision_image_time = time.monotonic()
        if self.vision_show_window:
            try:
                frame = self._ros_image_to_bgr_array(msg)
            except Exception as exc:
                self.get_logger().warn(f'Could not convert vision preview image: {exc}')
                return
            with self._vision_display_lock:
                self._vision_latest_display_image = frame
                if time.time() >= self._vision_display_hold_until:
                    self._vision_display_status = 'camera preview'

    def _vision_display_loop(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            self.get_logger().error(
                'vision_show_window is enabled but OpenCV/Numpy is not available. '
                f'Install python3-opencv python3-numpy. Details: {exc}'
            )
            return

        try:
            cv2.namedWindow(self.vision_window_name, cv2.WINDOW_NORMAL)
            while rclpy.ok() and not self._vision_window_shutdown.is_set():
                frame = self._vision_display_frame(np)
                cv2.imshow(self.vision_window_name, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q')):
                    self._vision_window_shutdown.set()
                    break
            cv2.destroyWindow(self.vision_window_name)
        except Exception as exc:
            self.get_logger().error(f'Qwen-Omni vision display window failed: {exc}')

    def _vision_display_frame(self, np_module):
        now = time.time()
        with self._vision_display_lock:
            latest = (
                None
                if self._vision_latest_display_image is None
                else self._vision_latest_display_image.copy()
            )
            display = (
                None
                if self._vision_display_image is None
                else self._vision_display_image.copy()
            )
            status = self._vision_display_status
            hold_active = now < self._vision_display_hold_until

        if display is not None and hold_active:
            frame = display
        elif latest is not None:
            frame = latest
            status = status or 'camera preview'
        else:
            frame = np_module.zeros((480, 640, 3), dtype=np_module.uint8)
            status = status or 'waiting for camera image...'

        if status:
            self._draw_status_bar(frame, status)
        return frame

    def _set_vision_display_result(self, image, status: str) -> None:
        with self._vision_display_lock:
            self._vision_display_image = image.copy()
            self._vision_display_status = status
            self._vision_display_hold_until = time.time() + max(0.0, self.vision_result_hold_sec)

    def _set_vision_display_status(self, status: str) -> None:
        with self._vision_display_lock:
            self._vision_display_status = status

    @staticmethod
    def _ascii_for_cv_text(text: str, fallback: str = '') -> str:
        value = str(text or '')
        encoded = value.encode('ascii', errors='ignore').decode('ascii')
        encoded = ' '.join(encoded.split())
        return encoded or fallback

    @staticmethod
    def _draw_status_bar(image, text: str) -> None:
        try:
            import cv2
        except ImportError:
            return
        if image is None or getattr(image, 'size', 0) == 0:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        text = McpOmniClient._ascii_for_cv_text(text, 'Qwen-Omni vision')
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

    def _save_and_show_api_detection_result(
        self,
        detections: List[Dict],
        answer,
        mode: str,
        status: str,
    ) -> str:
        try:
            image_msg = self._latest_camera_image_msg()
            bgr = self._ros_image_to_bgr_array(image_msg)
            result_bgr = self._draw_api_detections(bgr, detections)
            if self.vision_show_window:
                self._set_vision_display_result(result_bgr, status)
            return self._save_vision_outputs(
                image_payload=self._ros_image_to_jpeg_payload(image_msg),
                answer=answer,
                boxes=detections,
                mode=mode,
                annotated_bgr=result_bgr,
            )
        except Exception as exc:
            self.get_logger().warn(f'Could not save/show API detection result: {exc}')
            return ''

    def _save_vision_outputs(
        self,
        image_payload: Dict,
        answer,
        boxes: List[Dict],
        mode: str,
        annotated_bgr=None,
    ) -> str:
        if not self.vision_save_images:
            return ''
        try:
            self.vision_output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            prefix = self.vision_output_dir / f'{timestamp}_{uuid.uuid4().hex[:6]}'
            raw_path = prefix.with_name(prefix.name + '_raw.jpg')
            result_json_path = prefix.with_name(prefix.name + '_result.json')
            with open(raw_path, 'wb') as file_obj:
                file_obj.write(base64.b64decode(image_payload['jpeg_b64']))
            annotated_path = ''
            if annotated_bgr is not None:
                annotated_path = str(prefix.with_name(prefix.name + '_annotated.jpg'))
                self._write_bgr_image(annotated_path, annotated_bgr)
            result = {
                'mode': mode,
                'raw_image': str(raw_path),
                'annotated_image': annotated_path,
                'answer': answer,
                'boxes': boxes,
                'vision_image_topic': self.vision_image_topic,
                'image_width': image_payload.get('original_width'),
                'image_height': image_payload.get('original_height'),
                'api_width': image_payload.get('api_width'),
                'api_height': image_payload.get('api_height'),
                'stamp_sec': image_payload.get('stamp_sec'),
                'stamp_nanosec': image_payload.get('stamp_nanosec'),
                'frame_id': image_payload.get('frame_id'),
            }
            with open(result_json_path, 'w', encoding='utf-8') as file_obj:
                json.dump(result, file_obj, ensure_ascii=False, indent=2)
            return str(result_json_path)
        except Exception as exc:
            self.get_logger().warn(f'Could not save Omni vision outputs: {exc}')
            return ''

    def _draw_api_detections(self, bgr, detections: List[Dict]):
        try:
            import cv2
        except ImportError:
            return bgr
        image = bgr.copy()
        height, width = image.shape[:2]
        for index, detection in enumerate(detections):
            bbox = detection.get('bbox_xyxy', [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            color = self._box_color(index)
            label = str(detection.get('class_name') or 'object')
            confidence = detection.get('confidence', None)
            if confidence is not None:
                try:
                    label = f'{label} {float(confidence):.2f}'
                except (TypeError, ValueError):
                    pass
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            label_y1 = max(0, y1 - th - baseline - 6)
            label_y2 = min(height - 1, label_y1 + th + baseline + 6)
            label_x2 = min(width - 1, x1 + tw + 8)
            cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 4, label_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            center = detection.get('center_xy_int') or detection.get('center_xy')
            if center and len(center) == 2:
                cx = int(round(float(center[0])))
                cy = int(round(float(center[1])))
                if 0 <= cx < width and 0 <= cy < height:
                    cv2.drawMarker(
                        image,
                        (cx, cy),
                        (255, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=24,
                        thickness=4,
                        line_type=cv2.LINE_AA,
                    )
                    cv2.drawMarker(
                        image,
                        (cx, cy),
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=24,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
        return image

    @staticmethod
    def _box_color(index: int) -> Tuple[int, int, int]:
        palette = [
            (255, 56, 56),
            (255, 157, 151),
            (255, 112, 31),
            (255, 178, 29),
            (72, 249, 10),
            (0, 212, 187),
            (44, 153, 168),
            (100, 115, 255),
        ]
        return palette[index % len(palette)]

    def _latest_camera_jpeg_data_url(self) -> str:
        return self._latest_camera_jpeg_payload()['data_url']

    def _latest_camera_jpeg_payload_or_none(self) -> Optional[Dict]:
        if not self.vision_enabled:
            return None
        try:
            payload = self._latest_camera_jpeg_payload()
        except Exception as exc:
            self.get_logger().warn(f'Could not attach latest camera image to Omni turn: {exc}')
            return None
        self._publish_status(
            f'omni_image_attached topic={self.vision_image_topic} '
            f'api_size={payload["api_width"]}x{payload["api_height"]} '
            f'original_size={payload["original_width"]}x{payload["original_height"]}'
        )
        return payload

    def _latest_camera_jpeg_payload(self) -> Dict:
        image_msg = self._latest_camera_image_msg()
        return self._ros_image_to_jpeg_payload(image_msg)

    def _latest_camera_image_msg(self) -> RosImage:
        with self._vision_image_lock:
            image_msg = self._latest_vision_image
            image_time = self._latest_vision_image_time

        if image_msg is None:
            raise RuntimeError(f'No camera image received on {self.vision_image_topic}.')
        age = time.monotonic() - image_time
        if self.vision_image_max_age_sec > 0 and age > self.vision_image_max_age_sec:
            raise RuntimeError(
                f'Latest camera image is stale: {age:.1f}s old on {self.vision_image_topic}.'
            )
        return image_msg

    def _ros_image_to_jpeg_data_url(self, msg: RosImage) -> str:
        return self._ros_image_to_jpeg_payload(msg)['data_url']

    def _ros_image_to_jpeg_payload(self, msg: RosImage) -> Dict:
        image = self._ros_image_to_pil(msg)
        original_width = image.width
        original_height = image.height
        if self.vision_max_image_width > 0 and original_width > self.vision_max_image_width:
            ratio = self.vision_max_image_width / float(original_width)
            new_size = (self.vision_max_image_width, max(1, int(original_height * ratio)))
            from PIL import Image as PilImage
            resampling = getattr(getattr(PilImage, 'Resampling', PilImage), 'LANCZOS')
            image = image.resize(new_size, resampling)

        buffer = io.BytesIO()
        quality = max(1, min(95, self.vision_jpeg_quality))
        image.save(buffer, format='JPEG', quality=quality)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
        return {
            'jpeg_b64': image_b64,
            'data_url': f'data:image/jpeg;base64,{image_b64}',
            'api_width': image.width,
            'api_height': image.height,
            'original_width': original_width,
            'original_height': original_height,
            'stamp_sec': int(msg.header.stamp.sec),
            'stamp_nanosec': int(msg.header.stamp.nanosec),
            'frame_id': str(msg.header.frame_id),
        }

    def _ros_image_to_bgr_array(self, msg: RosImage):
        try:
            import numpy as np
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                'OpenCV/Numpy is not installed. Install python3-opencv python3-numpy.'
            ) from exc

        encoding = str(msg.encoding).lower()
        width = int(msg.width)
        height = int(msg.height)
        channel_map = {
            'rgb8': 3,
            'bgr8': 3,
            'rgba8': 4,
            'bgra8': 4,
            'mono8': 1,
            '8uc1': 1,
            '8uc3': 3,
        }
        if encoding not in channel_map:
            raise RuntimeError(
                f'Unsupported camera image encoding for display: {msg.encoding}. '
                'Use a color topic such as /camera/camera/color/image_raw.'
            )
        channels = channel_map[encoding]
        expected_step = width * channels
        packed = self._pack_image_rows(bytes(msg.data), height, int(msg.step), expected_step)
        if channels == 1:
            image = np.frombuffer(packed, dtype=np.uint8).reshape((height, width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.frombuffer(packed, dtype=np.uint8).reshape((height, width, channels)).copy()
        if encoding in ('rgb8', '8uc3'):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == 'bgr8':
            return image
        if encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image[:, :, :3]

    @staticmethod
    def _write_bgr_image(path: str, image) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('OpenCV is not installed; cannot save annotated image.') from exc
        if not cv2.imwrite(path, image):
            raise RuntimeError(f'cv2.imwrite failed: {path}')

    def _ros_image_to_pil(self, msg: RosImage):
        try:
            from PIL import Image as PilImage
        except ImportError as exc:
            raise RuntimeError(
                'Python package Pillow is not installed. Install it with: '
                '/usr/bin/python3 -m pip install --user -U pillow'
            ) from exc

        encoding = str(msg.encoding).lower()
        raw_modes = {
            'rgb8': ('RGB', 'RGB', 3),
            'bgr8': ('RGB', 'BGR', 3),
            'rgba8': ('RGBA', 'RGBA', 4),
            'bgra8': ('RGBA', 'BGRA', 4),
            'mono8': ('L', 'L', 1),
            '8uc1': ('L', 'L', 1),
            '8uc3': ('RGB', 'BGR', 3),
        }
        if encoding not in raw_modes:
            raise RuntimeError(
                f'Unsupported camera image encoding for vision: {msg.encoding}. '
                'Use a color topic such as /camera/camera/color/image_raw.'
            )

        mode, raw_mode, channels = raw_modes[encoding]
        width = int(msg.width)
        height = int(msg.height)
        expected_step = width * channels
        packed = self._pack_image_rows(bytes(msg.data), height, int(msg.step), expected_step)
        image = PilImage.frombytes(mode, (width, height), packed, 'raw', raw_mode)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    @staticmethod
    def _pack_image_rows(data: bytes, height: int, step: int, expected_step: int) -> bytes:
        if step == expected_step:
            return data[:height * expected_step]
        rows = []
        for row in range(height):
            start = row * step
            rows.append(data[start:start + expected_step])
        return b''.join(rows)

    def _api_key_from_env(self, env_name: str) -> str:
        api_key = os.environ.get(env_name, '').strip()
        if not api_key:
            raise RuntimeError(f'Environment variable {env_name} is not set.')
        try:
            api_key.encode('ascii')
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                f'Environment variable {env_name} contains non-ASCII text. '
                'Use the real API key, not a Chinese placeholder.'
            ) from exc
        return api_key

    def _send_omni_function_result(self, conversation, function_call: Dict, result_text: str) -> None:
        call_id = str(function_call.get('call_id') or '').strip()
        if not call_id:
            return
        item = {
            'id': f'item_{uuid.uuid4().hex}',
            'type': 'function_call_output',
            'call_id': call_id,
            'output': result_text,
        }
        if hasattr(conversation, 'create_item'):
            conversation.create_item(item)
            return
        if hasattr(conversation, 'send_event'):
            conversation.send_event({
                'type': 'conversation.item.create',
                'item': item,
            })
            return
        raise RuntimeError(
            'DashScope Realtime SDK does not expose create_item/send_event for tool output. '
            'Upgrade dashscope to >= 1.23.9.'
        )

    def _stream_pcm_to_realtime(self, conversation, pcm_bytes: bytes) -> None:
        chunk_size = max(320, self.omni_realtime_chunk_bytes)
        for index in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[index:index + chunk_size]
            conversation.append_audio(base64.b64encode(chunk).decode('ascii'))
            if self.omni_realtime_chunk_sleep_sec > 0:
                time.sleep(self.omni_realtime_chunk_sleep_sec)

    def _wait_realtime_done(self, done_event, error_text: Dict[str, str], label: str) -> None:
        if not done_event.wait(timeout=max(1.0, self.omni_timeout)):
            raise RuntimeError(f'{label} timed out.')
        if error_text['value']:
            error = error_text['value']
            error_text['value'] = ''
            raise RuntimeError(f'{label} error: {error}')

    @staticmethod
    def _clear_realtime_buffers(done_event, final_text, text_parts, function_calls, lock) -> None:
        with lock:
            text_parts.clear()
        final_text['value'] = ''
        function_calls.clear()
        done_event.clear()

    @staticmethod
    def _collected_realtime_text(final_text, text_parts, lock) -> str:
        text = final_text['value'].strip()
        if text:
            return text
        with lock:
            return ''.join(text_parts).strip()

    def _wav_to_realtime_pcm(self, wav_path: Path) -> bytes:
        with wave.open(str(wav_path), 'rb') as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            frames = wav.getnframes()
            pcm_bytes = wav.readframes(frames)
        if channels != 1 or sample_width != 2 or frame_rate != 16000:
            raise RuntimeError(
                'Qwen-Omni Realtime requires raw PCM 16kHz mono 16-bit audio. '
                f'Current WAV is channels={channels}, sample_width={sample_width}, '
                f'sample_rate={frame_rate}. Set sample_rate: 16000, channels: 1, '
                'record_format: S16_LE in mcp_omni_client.yaml.'
            )
        if not pcm_bytes:
            raise RuntimeError(f'No audio data in {wav_path}.')
        return pcm_bytes

    @staticmethod
    def _extract_realtime_done_text(response: dict) -> str:
        response_obj = response.get('response', {})
        output = response_obj.get('output', [])
        parts = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get('content', [])
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get('text') or content_item.get('transcript')
                    if text:
                        parts.append(str(text).strip())
        return '\n'.join(part for part in parts if part).strip()

    @staticmethod
    def _collect_compatible_stream_response(response):
        text_parts: List[str] = []
        tool_calls_by_index: Dict[int, Dict] = {}
        for chunk in response:
            choices = getattr(chunk, 'choices', None) or []
            if not choices:
                continue
            delta = getattr(choices[0], 'delta', None)
            if delta is None:
                continue
            content = getattr(delta, 'content', None)
            if isinstance(content, str):
                text_parts.append(content)
            tool_call_deltas = getattr(delta, 'tool_calls', None) or []
            for tool_call_delta in tool_call_deltas:
                index = int(getattr(tool_call_delta, 'index', len(tool_calls_by_index)))
                entry = tool_calls_by_index.setdefault(
                    index,
                    {'id': '', 'name': '', 'arguments': ''},
                )
                call_id = getattr(tool_call_delta, 'id', None)
                if call_id:
                    entry['id'] = str(call_id)
                function = getattr(tool_call_delta, 'function', None)
                if function is None:
                    continue
                name = getattr(function, 'name', None)
                if name:
                    entry['name'] += str(name)
                arguments = getattr(function, 'arguments', None)
                if arguments:
                    entry['arguments'] += str(arguments)

        tool_calls = []
        for index in sorted(tool_calls_by_index):
            entry = tool_calls_by_index[index]
            if entry['name']:
                if not entry['id']:
                    entry['id'] = f'call_{index}'
                tool_calls.append(entry)
        return ''.join(text_parts).strip(), tool_calls

    def _tool_result_failed(self, result_text: str) -> bool:
        lowered = result_text.lower()
        failure_markers = (
            ' failed:',
            ' rejected:',
            ' unavailable',
            ' timed out',
            'unsupported omni tool',
        )
        return any(marker in lowered for marker in failure_markers)

    def _format_failed_tool_answer(self, failed_results: List[str]) -> str:
        return '本次命令没有执行成功：' + '；'.join(failed_results)

    @staticmethod
    def _silence_pcm_bytes(byte_count: int) -> bytes:
        if byte_count % 2:
            byte_count += 1
        return b'\x00' * byte_count


def main(args=None) -> None:
    rclpy.init(args=args)
    node = McpOmniClient()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

import base64
import json
import os
import re
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from audio_dialog.audio_dialog_node import AudioDialogNode
from mcp.srv import CallTool, ListTools
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import Trigger


class McpOmniClient(AudioDialogNode):
    """Qwen3.5-Omni Realtime multimodal MCP client."""

    def __init__(self) -> None:
        super().__init__('mcp_omni_client')

        self._tool_schema_cache = None
        self._tool_schema_cache_time = 0.0
        self._conversation_memory_lock = threading.Lock()
        self._conversation_memory: List[Dict] = []
        self._conversation_tool_result_lock = threading.Lock()
        self._current_conversation_tool_results: Optional[List[str]] = None

        self.list_tools_client = self.create_client(ListTools, self.list_tools_service)
        self.call_tool_client = self.create_client(CallTool, self.call_tool_service)
        self.emergency_stop_client = self.create_client(
            Trigger,
            self.mcp_emergency_stop_service,
        )

        self.get_logger().info(
            'Qwen-Omni Realtime MCP client ready. '
            f'model={self.omni_model}, voice={self.omni_realtime_voice}, '
            f'realtime_url={self.omni_realtime_url}, '
            f'list_tools_service={self.list_tools_service}, '
            f'call_tool_service={self.call_tool_service}.'
        )

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
        self.declare_parameter('conversation_memory_enabled', True)
        self.declare_parameter('conversation_memory_max_turns', 8)
        self.declare_parameter('conversation_memory_max_chars', 4000)
        self.declare_parameter('omni_native_audio_output_enabled', True)
        self.declare_parameter('omni_native_audio_fallback_to_local_tts', False)
        self.declare_parameter('omni_native_audio_sample_rate', 24000)
        self.declare_parameter('omni_native_audio_silence_bytes', 3200)
        self.declare_parameter('omni_native_audio_timeout', 90.0)
        self.declare_parameter('omni_speech_rate', 'normal')
        self.declare_parameter('omni_speech_volume', 'normal')
        self.declare_parameter('omni_speech_emotion', 'natural')
        self.declare_parameter('omni_speech_style', '清晰、自然、友好，适合机器人语音助手')

        self.declare_parameter('list_tools_service', '/mcp_server/list_tools')
        self.declare_parameter('call_tool_service', '/mcp_server/call_tool')
        self.declare_parameter('mcp_emergency_stop_service', '/mcp_server/emergency_stop')
        self.declare_parameter('mcp_service_wait_timeout_sec', 10.0)
        self.declare_parameter('mcp_tool_timeout_sec', 80.0)
        self.declare_parameter('mcp_long_tool_timeout_sec', 900.0)
        self.declare_parameter('mcp_max_tool_rounds', 10)

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
        self.conversation_memory_enabled = self._as_bool(
            self.get_parameter('conversation_memory_enabled').value
        )
        self.conversation_memory_max_turns = max(
            1,
            int(self.get_parameter('conversation_memory_max_turns').value),
        )
        self.conversation_memory_max_chars = max(
            500,
            int(self.get_parameter('conversation_memory_max_chars').value),
        )
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

        self.list_tools_service = str(self.get_parameter('list_tools_service').value)
        self.call_tool_service = str(self.get_parameter('call_tool_service').value)
        self.mcp_emergency_stop_service = str(
            self.get_parameter('mcp_emergency_stop_service').value
        )
        self.mcp_service_wait_timeout_sec = float(
            self.get_parameter('mcp_service_wait_timeout_sec').value
        )
        self.mcp_tool_timeout_sec = float(self.get_parameter('mcp_tool_timeout_sec').value)
        self.mcp_long_tool_timeout_sec = float(
            self.get_parameter('mcp_long_tool_timeout_sec').value
        )
        self.mcp_max_tool_rounds = max(1, int(self.get_parameter('mcp_max_tool_rounds').value))

    def _process_audio_file(self, wav_path: Path):
        self._publish_status(f'omni_audio_input={wav_path}')
        self._set_last_transcript_text('')
        self.transcript_pub.publish(String(data=''))
        self._begin_conversation_tool_capture()
        try:
            response_text = self._run_omni_realtime_tool_turn(wav_path)
        finally:
            tool_results = self._finish_conversation_tool_capture()
        transcript = self._get_last_transcript_text() or f'语音输入:{wav_path.name}'
        self._remember_conversation_turn(transcript, response_text, tool_results)
        self._publish_response(response_text)
        self._speak_omni_response(response_text)
        return True, response_text

    def _process_text(self, text: str, source: str = 'text'):
        if not text:
            text = self.no_speech_text or '没有收到文本，请再输入一次。'
            self._set_last_transcript_text('')
            self._publish_status('transcript=')
            self.transcript_pub.publish(String(data=''))
            self._publish_response(text)
            self._speak_omni_response(text)
            return True, text

        self._set_last_transcript_text(text)
        self._publish_status(f'transcript={text}')
        self.transcript_pub.publish(String(data=text))
        self._begin_conversation_tool_capture()
        try:
            response_text = self._run_omni_text_tool_turn(text)
        finally:
            tool_results = self._finish_conversation_tool_capture()
        self._remember_conversation_turn(text, response_text, tool_results)
        self._publish_response(response_text)
        self._speak_omni_response(response_text)
        return True, response_text

    def _text_popup_actions(self):
        return [
            ('急停', self._run_emergency_stop_popup_action, {
                'danger': True,
                'always_enabled': True,
            }),
            ('关闭夹爪', lambda: self._run_popup_action('close_gripper', '关闭夹爪')),
            ('打开夹爪', lambda: self._run_popup_action('open_gripper', '打开夹爪')),
            ('回到初始位置', lambda: self._run_popup_action('go_home', '回到初始位置')),
        ]

    def _run_emergency_stop_popup_action(self) -> str:
        self._interrupt_tts_playback()
        self._publish_status('text_popup_action=emergency_stop')
        if not self.emergency_stop_client.wait_for_service(timeout_sec=1.0):
            message = f'{self.mcp_emergency_stop_service} unavailable'
            self._publish_response(f'急停失败：{message}')
            return f'急停失败：{message}'

        future = self.emergency_stop_client.call_async(Trigger.Request())
        response = self._wait_for_service_future(
            future,
            self.mcp_emergency_stop_service,
            timeout_sec=2.0,
        )
        if response is None:
            message = '急停请求已发出，但等待服务反馈超时。请观察机械臂状态。'
            self._publish_response(message)
            return message

        status = '已触发' if response.success else '失败'
        message = str(response.message or '').strip()
        response_text = f'急停{status}：{message or "无返回信息"}'
        self._capture_conversation_tool_result(response_text)
        self._publish_response(response_text)
        return response_text

    def _run_popup_action(self, tool_name: str, label: str) -> str:
        if not self._busy_lock.acquire(blocking=False):
            return '现在还在处理上一条消息，请稍等一下。'
        try:
            self._publish_status(f'text_popup_action={tool_name}')
            self._begin_conversation_tool_capture()
            try:
                response_text = self._run_omni_named_tool(tool_name, {})
            finally:
                tool_results = self._finish_conversation_tool_capture()
            self._remember_conversation_turn(label, response_text, tool_results)
            self._publish_response(response_text)
            self._speak_omni_response(response_text)
            if self._tool_result_failed(response_text):
                return f'{label}失败：{response_text}'
            return f'{label}已执行：{response_text}'
        finally:
            self._busy_lock.release()

    def _run_omni_realtime_tool_turn(self, wav_path: Path) -> str:
        try:
            import dashscope
            from openai import OpenAI
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
        route_client = OpenAI(
            api_key=self._api_key_from_env(self.omni_api_key_env),
            base_url=self.omni_base_url,
            timeout=self.omni_timeout,
        )
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
                        node._set_last_transcript_text(user_transcript['value'])
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
                'instructions': self._realtime_tool_instructions_with_memory(),
            }
            if self.omni_realtime_enable_search:
                update_kwargs['enable_search'] = True
                update_kwargs['search_options'] = {'enable_source': True}
            if self.omni_max_tokens > 0:
                update_kwargs['max_tokens'] = self.omni_max_tokens
            tools = self._omni_tool_schema()
            if tools:
                update_kwargs['tools'] = tools
            conversation.update_session(**update_kwargs)

            self._stream_pcm_to_realtime(conversation, pcm_bytes)
            conversation.commit()
            conversation.create_response()
            self._wait_realtime_done(done_event, error_text, 'Qwen-Omni Realtime response')

            all_tool_results = []
            for _ in range(self.mcp_max_tool_rounds):
                if not function_calls:
                    answer = self._collected_realtime_text(final_text, text_parts, lock)
                    direct_calls = self._direct_tool_calls_from_user_text(user_transcript['value'])
                    if direct_calls:
                        self._publish_status(
                            'omni_realtime_direct_tool_route='
                            + ';'.join(
                                f'{call["name"]} args={call["arguments"]}'
                                for call in direct_calls
                            )
                        )
                        tool_results = []
                        for direct_call in direct_calls:
                            result_text = self._run_compatible_tool_call(direct_call)
                            tool_results.append(result_text)
                            all_tool_results.append(result_text)
                            if self._tool_result_failed(result_text):
                                break
                        failed_results = [
                            text for text in tool_results if self._tool_result_failed(text)
                        ]
                        if failed_results:
                            return self._format_failed_tool_answer(
                                user_transcript['value'],
                                failed_results,
                            )
                        if self._tool_results_complete_user_request(
                            user_transcript['value'],
                            tool_results,
                        ):
                            return self._format_success_tool_answer(
                                user_transcript['value'],
                                tool_results,
                            )
                        return self._guard_final_answer(
                            '；'.join(tool_results) or answer,
                            all_tool_results,
                            user_transcript['value'],
                        )
                    routed_call = self._route_next_text_step_with_llm(
                        route_client,
                        user_transcript['value'],
                        answer,
                        all_tool_results,
                    )
                    if routed_call:
                        if routed_call['name'] == 'final_answer':
                            final_answer = self._final_answer_from_routed_call(routed_call) or answer
                            return self._guard_final_answer(
                                final_answer,
                                all_tool_results,
                                user_transcript['value'],
                            )
                        self._publish_status(
                            'omni_realtime_tool_router='
                            f'{routed_call["name"]} args={routed_call["arguments"]}'
                        )
                        result_text = self._run_compatible_tool_call(routed_call)
                        all_tool_results.append(result_text)
                        if self._tool_result_failed(result_text):
                            return self._format_failed_tool_answer(
                                user_transcript['value'],
                                [result_text],
                            )
                        return self._guard_final_answer(
                            result_text,
                            all_tool_results,
                            user_transcript['value'],
                        )
                    return self._guard_final_answer(
                        answer,
                        all_tool_results,
                        user_transcript['value'],
                    )

                current_calls = list(function_calls)
                tool_results = []
                for function_call in current_calls:
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
                    return self._format_failed_tool_answer(
                        user_transcript['value'],
                        failed_results,
                    )
                if self._tool_results_complete_user_request(
                    user_transcript['value'],
                    tool_results,
                ):
                    return self._format_success_tool_answer(
                        user_transcript['value'],
                        tool_results,
                    )

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
                user_transcript['value'],
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
        direct_calls = self._direct_tool_calls_from_user_text(user_text)
        if direct_calls:
            self._publish_status(
                'omni_pre_direct_tool_route='
                + ';'.join(
                    f'{call["name"]} args={call["arguments"]}' for call in direct_calls
                )
            )
            return self._run_direct_tool_calls_for_request(user_text, direct_calls)

        messages = [{'role': 'system', 'content': self._omni_tool_instructions()}]
        messages.extend(self._conversation_context_messages())
        messages.append({'role': 'user', 'content': self._omni_user_content(user_text)})
        self._publish_status(f'llm=omni_text_tools model={self.omni_text_model}')
        all_tool_results = []
        for _ in range(self.mcp_max_tool_rounds):
            request_kwargs = {
                'model': self.omni_text_model,
                'messages': messages,
                'modalities': ['text'],
                'stream': True,
                'stream_options': {'include_usage': True},
                'max_tokens': self.omni_max_tokens,
            }
            tools = self._omni_tool_schema()
            if tools:
                request_kwargs['tools'] = tools
                request_kwargs['tool_choice'] = 'auto'
            response = client.chat.completions.create(**request_kwargs)
            answer, tool_calls = self._collect_compatible_stream_response(response)
            if not tool_calls:
                direct_calls = self._direct_tool_calls_from_user_text(user_text)
                if direct_calls:
                    tool_calls = direct_calls
                    self._publish_status(
                        'omni_direct_tool_route='
                        + ';'.join(
                            f'{call["name"]} args={call["arguments"]}' for call in direct_calls
                        )
                    )
                else:
                    pseudo_call = self._pseudo_tool_call_from_answer(answer, user_text)
                    if pseudo_call:
                        tool_calls = [pseudo_call]
                        self._publish_status(
                            'omni_pseudo_tool_route='
                            f'{pseudo_call["name"]} args={pseudo_call["arguments"]}'
                        )
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
                        return self._guard_final_answer(final_answer, all_tool_results, user_text)
                    tool_calls = [routed_call]
                    self._publish_status(
                        'omni_tool_router='
                        f'{routed_call["name"]} args={routed_call["arguments"]}'
                    )
                elif answer:
                    return self._guard_final_answer(answer, all_tool_results, user_text)
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
                return self._format_failed_tool_answer(user_text, failed_results)
            if self._tool_results_complete_user_request(user_text, tool_results):
                return self._format_success_tool_answer(user_text, tool_results)

            messages.append({
                'role': 'system',
                'content': self._post_tool_instructions(user_text, tool_results),
            })

        self.get_logger().warn('Qwen-Omni requested too many text tool rounds; stopping tool loop.')
        return self._guard_final_answer(
            '；'.join(all_tool_results) or '好的。',
            all_tool_results,
            user_text,
        )

    def _run_direct_tool_calls_for_request(self, user_text: str, direct_calls: List[Dict]) -> str:
        tool_results = []
        all_tool_results = []
        for direct_call in direct_calls:
            result_text = self._run_compatible_tool_call(direct_call)
            tool_results.append(result_text)
            all_tool_results.append(result_text)
            if self._tool_result_failed(result_text):
                break

        failed_results = [text for text in tool_results if self._tool_result_failed(text)]
        if failed_results:
            return self._format_failed_tool_answer(user_text, failed_results)
        if self._tool_results_complete_user_request(user_text, tool_results):
            return self._format_success_tool_answer(user_text, tool_results)
        if tool_results and all(
            self._is_terminal_action_success_result(text) for text in tool_results
        ):
            return self._format_success_tool_answer(user_text, tool_results)
        return self._guard_final_answer(
            '；'.join(tool_results) or '工具调用没有返回结果。',
            all_tool_results,
            user_text,
        )

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

    def _guard_final_answer(
        self,
        answer: str,
        tool_results: List[str],
        user_text: str = '',
    ) -> str:
        answer = str(answer or '').strip()
        if not answer:
            return ''
        if (
            self._looks_like_execution_request(user_text)
            and not tool_results
            and self._claims_action_success(answer)
        ):
            return '这是一条新的执行命令，我还没有调用工具，不能确认已经完成。'
        if self._claims_grasp_success(answer) and not any(
            self._is_grab_success_result(result) for result in tool_results
        ):
            if any(self._is_grab_attempt_result(result) for result in tool_results):
                failed_results = [
                    result for result in tool_results
                    if self._is_grab_attempt_result(result) and self._tool_result_failed(result)
                ]
                if failed_results:
                    return self._format_failed_tool_answer('当前抓取请求', failed_results)
                return '抓取命令已经发送，但还没有确认真正抓取完成。'
            return '我还没有执行抓取工具，不能确认抓取成功。'
        if any(self._is_grab_success_result(result) for result in tool_results):
            return '已发送抓取命令，正在执行抓取。'
        return answer

    def _looks_like_execution_request(self, user_text: str) -> bool:
        text = ''.join(str(user_text or '').strip().split()).lower()
        if not text or self._looks_like_question(text):
            return False
        action_tokens = (
            '把',
            '将',
            '放',
            '放在',
            '放到',
            '放进',
            '放入',
            '抓',
            '抓取',
            '拿',
            '拿起',
            '夹',
            '夹取',
            '打开夹爪',
            '关闭夹爪',
            '回到',
            'home',
            '移动',
            '下降',
            '上升',
        )
        return any(token in text for token in action_tokens)

    @staticmethod
    def _claims_action_success(answer: str) -> bool:
        text = ''.join(str(answer or '').strip().split()).lower()
        if not text:
            return False
        success_markers = (
            '已成功',
            '已经成功',
            '成功将',
            '成功把',
            '已完成',
            '已经完成',
            '任务完成',
            '放好了',
            '抓好了',
            '完成了',
            'successfully',
            'completed',
        )
        return any(marker in text for marker in success_markers)

    def _route_next_text_step_with_llm(
        self,
        client,
        user_text: str,
        draft_answer: str,
        tool_results: List[str],
    ) -> Optional[Dict]:
        if not tool_results and not draft_answer:
            return None
        available_tools = self._tool_summaries_for_router()
        tool_names = ['final_answer'] + [tool['name'] for tool in available_tools]
        if len(tool_names) <= 1:
            return None
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是机器人工具路由器。你只判断下一步应该是最终回答还是继续调用一个工具。'
                    '必须基于用户真实意图、已有工具结果和助手草稿，不要做关键词截取。'
                    '可用工具的能力边界、参数和语义全部来自 available_tools。'
                    '如果用户请求依赖当前相机画面、实时环境、当前可见物体、当前机器人/夹爪状态，'
                    '或助手草稿表示需要调用工具才能确认，而 available_tools 里有匹配工具，'
                    '必须选择最匹配的工具，不要选择 final_answer，也不要要求用户再次确认。'
                    '如果用户是在要求立即执行一个可用工具能力，也应选择对应工具。'
                    '只有不需要实时信息、也不需要执行工具的知识性问题、能力确认、可行性询问或聊天，'
                    '才选择 final_answer。'
                    '如果用户要求的任务没有对应工具，不要用相似工具代替，应选择 final_answer 并说明限制。'
                    '只能输出 JSON，不要解释。'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'original_user_request': user_text,
                    'conversation_context': self._conversation_context_data(),
                    'assistant_draft_answer': draft_answer,
                    'tool_results_so_far': tool_results,
                    'available_actions': tool_names,
                    'available_tools': available_tools,
                    'output_schema': {
                        'action': 'final_answer or one available tool name',
                        'arguments': 'object matching the selected tool parameters',
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
        return {
            'id': f'call_router_{uuid.uuid4().hex}',
            'name': action,
            'arguments': json.dumps(arguments, ensure_ascii=False),
        }

    def _final_answer_from_routed_call(self, routed_call: Dict) -> str:
        try:
            arguments = json.loads(str(routed_call.get('arguments') or '{}'))
        except json.JSONDecodeError:
            return ''
        return str(arguments.get('final_answer') or '').strip()

    def _direct_tool_call_from_user_text(self, user_text: str) -> Optional[Dict]:
        calls = self._direct_tool_calls_from_user_text(user_text)
        return calls[0] if calls else None

    def _direct_tool_calls_from_user_text(self, user_text: str) -> List[Dict]:
        text = ''.join(str(user_text or '').strip().split())
        if not text:
            return []
        if self._looks_like_question(text):
            return []

        place_match = re.search(
            r'^(?:请|帮我|麻烦你)?(?:把|将)(.+?)(?:放到|放进|放入|放在|放)(.+)$',
            text,
        )
        if place_match:
            object_phrase = place_match.group(1)
            object_name = self._clean_object_phrase(object_phrase)
            place_target = place_match.group(2)
            relative = self._parse_relative_place_target(place_target)
            if object_name and relative:
                return [self._make_tool_call(
                    'pick_and_place_relative',
                    {
                        'object_name': object_name,
                        'reference_object_name': relative['reference_object_name'],
                        'direction': relative['direction'],
                        'distance_cm': relative['distance_cm'],
                    },
                    'direct_place_relative',
                )]
            container_name = self._clean_container_phrase(place_target)
            if object_name and container_name and self._looks_like_container_target(place_target):
                if self._looks_like_multi_object_task(object_phrase):
                    task_request = str(user_text or '').strip()
                    if self._is_daily_fruit_task_text(object_phrase):
                        task_request = self._daily_fruit_task_request(task_request)
                    return [self._make_tool_call(
                        'pick_all_fruits_into_container',
                        {
                            'container_name': container_name,
                            'task_request': task_request,
                        },
                        'direct_all_fruits_container',
                    )]
                object_names = self._split_object_list_phrase(object_phrase)
                if not object_names:
                    object_names = [object_name]
                return [
                    self._make_tool_call(
                        'pick_and_place_into_container',
                        {
                            'object_name': name,
                            'container_name': container_name,
                        },
                        'direct_place_container',
                    )
                    for name in object_names
                ]

        grab_match = re.search(r'^(?:请|帮我|麻烦你)?(?:抓取|抓|拿起|拿|夹取|夹)(.+)$', text)
        if grab_match and not any(token in text for token in ('放', '放到', '放进', '放入')):
            object_name = self._clean_object_phrase(grab_match.group(1))
            if object_name:
                return [self._make_tool_call(
                    'grab_api_object',
                    {'object_name': object_name},
                    'direct_grab',
                )]
        return []

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        question_markers = (
            '为什么',
            '怎么',
            '如何',
            '能不能',
            '可不可以',
            '可以吗',
            '行不行',
            '吗',
            '?',
            '？',
        )
        return any(marker in str(text or '') for marker in question_markers)

    def _pseudo_tool_call_from_answer(self, answer: str, user_text: str) -> Optional[Dict]:
        text = str(answer or '')
        if not text:
            return None
        tool_name = ''
        xml_match = re.search(r'<function=([A-Za-z_][A-Za-z0-9_]*)\s*>', text)
        if xml_match:
            tool_name = xml_match.group(1)
        if not tool_name:
            code_match = re.search(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', text)
            if code_match and 'tool' in text.lower():
                tool_name = code_match.group(1)
        if not tool_name:
            return None

        mapped = self._map_legacy_tool_name(tool_name, user_text)
        if mapped == 'pick_and_place_into_container':
            direct = self._direct_tool_call_from_user_text(user_text)
            if direct and direct.get('name') == 'pick_and_place_into_container':
                return direct
        if mapped in ('observe_scene', 'list_api_objects'):
            return self._make_tool_call(
                mapped,
                {'question': user_text},
                'pseudo_vision',
            )
        return None

    def _map_legacy_tool_name(self, tool_name: str, user_text: str) -> str:
        name = str(tool_name or '').strip()
        text = ''.join(str(user_text or '').split())
        if name in (
            'fr3_vision_detect_objects',
            'get_camera_image_and_detect_objects',
            'get_camera_image',
            'detect_objects',
        ):
            if any(token in text for token in ('放到', '放进', '放入', '放')) and any(
                token in text for token in ('箱', '盒', '筐', 'box', 'container')
            ):
                return 'pick_and_place_into_container'
            return 'observe_scene'
        return name

    def _make_tool_call(self, name: str, arguments: Dict, prefix: str) -> Dict:
        return {
            'id': f'call_{prefix}_{uuid.uuid4().hex}',
            'name': name,
            'arguments': json.dumps(arguments or {}, ensure_ascii=False),
        }

    def _parse_relative_place_target(self, text: str) -> Optional[Dict]:
        value = str(text or '').strip()
        if not value:
            return None
        value = re.sub(r'(?:的位置|的位置处|位置|处)$', '', value)
        direction_pattern = r'(左边|右边|前面|后面|後面|上面|下面|左方|右方|前方|后方|後方|上方|下方|左|右|前|后|後|上|下|left|right|front|forward|back|behind|up|down|above|below)'
        distance_pattern = (
            r'([0-9]+(?:\.[0-9]+)?|'
            r'[零〇一二两兩三四五六七八九十百半]+(?:点[零〇一二两兩三四五六七八九]+)?)'
        )
        match = re.search(
            rf'(.+?){direction_pattern}(?:(?:约|大约|大概)?{distance_pattern}(?:厘米|公分|cm|CM)?)?$',
            value,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                rf'(.+?)(?:约|大约|大概)?{distance_pattern}(?:厘米|公分|cm|CM){direction_pattern}$',
                value,
                flags=re.IGNORECASE,
            )
            if not match:
                return None
            reference = self._clean_reference_phrase(match.group(1))
            distance_cm = self._parse_distance_cm(match.group(2))
            direction = self._normalize_relative_direction(match.group(3))
        else:
            reference = self._clean_reference_phrase(match.group(1))
            direction = self._normalize_relative_direction(match.group(2))
            distance_cm = (
                self._parse_distance_cm(match.group(3))
                if match.group(3)
                else self._default_relative_distance_cm(direction)
            )
        if not reference or not direction or distance_cm is None:
            return None
        return {
            'reference_object_name': reference,
            'direction': direction,
            'distance_cm': distance_cm,
        }

    @staticmethod
    def _default_relative_distance_cm(direction: str) -> float:
        if direction == '上':
            return 1.0
        return 5.0

    @staticmethod
    def _parse_distance_cm(text: str) -> Optional[float]:
        value = ''.join(str(text or '').strip().split()).lower()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        value = value.replace('兩', '两')
        if value == '半':
            return 0.5
        if '点' in value:
            integer_text, decimal_text = value.split('点', 1)
            integer_value = McpOmniClient._parse_chinese_integer(integer_text)
            if integer_value is None:
                return None
            digit_map = {
                '零': '0',
                '〇': '0',
                '一': '1',
                '二': '2',
                '两': '2',
                '三': '3',
                '四': '4',
                '五': '5',
                '六': '6',
                '七': '7',
                '八': '8',
                '九': '9',
            }
            decimal_digits = ''.join(digit_map.get(char, '') for char in decimal_text)
            if len(decimal_digits) != len(decimal_text):
                return None
            return float(f'{integer_value}.{decimal_digits}')
        integer_value = McpOmniClient._parse_chinese_integer(value)
        return float(integer_value) if integer_value is not None else None

    @staticmethod
    def _parse_chinese_integer(text: str) -> Optional[int]:
        value = ''.join(str(text or '').strip().split()).replace('兩', '两')
        if not value:
            return 0
        digit_map = {
            '零': 0,
            '〇': 0,
            '一': 1,
            '二': 2,
            '两': 2,
            '三': 3,
            '四': 4,
            '五': 5,
            '六': 6,
            '七': 7,
            '八': 8,
            '九': 9,
        }
        if value in digit_map:
            return digit_map[value]
        if '百' in value:
            left, _, right = value.partition('百')
            hundreds = McpOmniClient._parse_chinese_integer(left) if left else 1
            remainder = McpOmniClient._parse_chinese_integer(right) if right else 0
            if hundreds is None or remainder is None:
                return None
            return hundreds * 100 + remainder
        if '十' in value:
            left, _, right = value.partition('十')
            tens = McpOmniClient._parse_chinese_integer(left) if left else 1
            ones = McpOmniClient._parse_chinese_integer(right) if right else 0
            if tens is None or ones is None:
                return None
            return tens * 10 + ones
        if all(char in digit_map for char in value):
            return int(''.join(str(digit_map[char]) for char in value))
        return None

    @staticmethod
    def _normalize_relative_direction(text: str) -> str:
        value = ''.join(str(text or '').strip().lower().split())
        mapping = {
            '左': '左',
            '左边': '左',
            '左方': '左',
            'left': '左',
            '右': '右',
            '右边': '右',
            '右方': '右',
            'right': '右',
            '前': '前',
            '前面': '前',
            '前方': '前',
            'front': '前',
            'forward': '前',
            '后': '后',
            '後': '后',
            '后面': '后',
            '後面': '后',
            '后方': '后',
            '後方': '后',
            'back': '后',
            'behind': '后',
            '上': '上',
            '上面': '上',
            '上方': '上',
            'up': '上',
            'above': '上',
            '下': '下',
            '下面': '下',
            '下方': '下',
            'down': '下',
            'below': '下',
        }
        return mapping.get(value, value)

    @staticmethod
    def _clean_reference_phrase(text: str) -> str:
        value = str(text or '').strip()
        for prefix in ('在', '到', '至'):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        for token in ('这个', '那个', '这颗', '那颗', '一个', '一颗', '当前', '识别到的'):
            value = value.replace(token, '')
        return value.strip(' ，,。.;；:：的')

    @staticmethod
    def _looks_like_container_target(text: str) -> bool:
        value = ''.join(str(text or '').strip().split()).lower()
        return any(token in value for token in ('箱', '盒', '筐', '篮', 'container', 'box', 'bin', 'basket')) or value.endswith(('里', '里面', '内', '中'))

    @staticmethod
    def _clean_object_phrase(text: str) -> str:
        value = str(text or '').strip()
        for token in ('这个', '那个', '这颗', '那颗', '一个', '一颗', '当前', '识别到的'):
            value = value.replace(token, '')
        return value.strip(' ，,。.;；:：的')

    def _split_object_list_phrase(self, text: str) -> List[str]:
        value = str(text or '').strip()
        if not value:
            return []
        if self._looks_like_multi_object_task(''.join(value.split())):
            return []
        parts = [
            part
            for part in re.split(r'(?:、|和|跟|与|以及|,|，|;|；|再(?:把)?|然后(?:把)?|接着(?:把)?)', value)
            if part
        ]
        names = []
        for part in parts:
            name = self._clean_object_phrase(part)
            if name and name not in names:
                names.append(name)
        return names if len(names) > 1 else []

    @staticmethod
    def _clean_container_phrase(text: str) -> str:
        value = str(text or '').strip()
        for token in ('这个', '那个', '一个', '当前', '识别到的'):
            value = value.replace(token, '')
        value = value.strip(' ，,。.;；:：的')
        for suffix in ('里面', '里边', '里', '内', '中'):
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                break
        value = value.strip(' ，,。.;；:：的')
        aliases = {
            '箱': '箱子',
            '盒': '盒子',
            '框': '箱子',
            '筐': '筐',
            '篮': '篮子',
            '篮筐': '篮子',
        }
        return aliases.get(value, value)

    @staticmethod
    def _is_daily_fruit_task_text(text: str) -> bool:
        compact = ''.join(str(text or '').split()).lower()
        return '水果' in compact or 'fruit' in compact

    def _daily_fruit_task_request(self, user_text: str) -> str:
        text = str(user_text or '').strip()
        hint = (
            '（任务口径：所有水果按日常食用水果理解，不按植物学果实理解；'
            '辣椒、青椒、甜椒、番茄、黄瓜、茄子等日常作为蔬菜或调味食材的物体不属于本任务目标。）'
        )
        if hint in text:
            return text
        return f'{text}{hint}'

    def _claims_grasp_success(self, answer: str) -> bool:
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

    def _is_grab_attempt_result(self, result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        return 'grab_api_object' in lowered

    def _is_grab_success_result(self, result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        success_marker = 'grab_api_object success:' in lowered
        return (
            success_marker
            and ' failed:' not in lowered
            and ' rejected:' not in lowered
            and ' unavailable' not in lowered
            and ' timed out' not in lowered
        )

    def _omni_user_content(self, user_text: str):
        return user_text

    def _conversation_context_data(self) -> List[Dict]:
        if not getattr(self, 'conversation_memory_enabled', False):
            return []
        with self._conversation_memory_lock:
            turns = list(self._conversation_memory[-self.conversation_memory_max_turns:])
        if not turns:
            return []

        max_chars = max(500, int(self.conversation_memory_max_chars))
        selected = []
        total_chars = 0
        for turn in reversed(turns):
            slim = {
                'user': str(turn.get('user') or ''),
                'assistant': str(turn.get('assistant') or ''),
            }
            tool_results = turn.get('tool_results') or []
            if tool_results:
                slim['tool_results'] = [str(result) for result in tool_results[:6]]
            encoded_len = len(json.dumps(slim, ensure_ascii=False))
            if selected and total_chars + encoded_len > max_chars:
                break
            selected.append(slim)
            total_chars += encoded_len
        selected.reverse()
        return selected

    def _conversation_context_messages(self) -> List[Dict]:
        context = self._conversation_context_data()
        if not context:
            return []
        content = (
            '以下是同一位用户在本节点里的近期对话和工具结果记忆。'
            '回答“刚刚/之前/你都/已经”等追问时要优先参考这些记忆；'
            '如果用户的新请求要求读取当前画面或执行动作，仍然按工具说明调用工具。'
            f'\n{json.dumps(context, ensure_ascii=False)}'
        )
        return [{'role': 'system', 'content': content}]

    def _remember_conversation_turn(
        self,
        user_text: str,
        assistant_text: str,
        tool_results: Optional[List[str]] = None,
    ) -> None:
        if not getattr(self, 'conversation_memory_enabled', False):
            return
        user_text = str(user_text or '').strip()
        assistant_text = str(assistant_text or '').strip()
        results = [
            self._trim_memory_text(str(result or '').strip(), 900)
            for result in (tool_results or [])
            if str(result or '').strip()
        ]
        if not user_text and not assistant_text and not results:
            return
        turn = {
            'time_sec': time.time(),
            'user': self._trim_memory_text(user_text, 500),
            'assistant': self._trim_memory_text(assistant_text, 700),
            'tool_results': results[-8:],
        }
        with self._conversation_memory_lock:
            self._conversation_memory.append(turn)
            max_turns = max(1, int(self.conversation_memory_max_turns))
            if len(self._conversation_memory) > max_turns:
                self._conversation_memory = self._conversation_memory[-max_turns:]

    @staticmethod
    def _trim_memory_text(text: str, max_chars: int) -> str:
        value = str(text or '').strip()
        if len(value) <= max_chars:
            return value
        return value[:max(0, max_chars - 3)].rstrip() + '...'

    def _begin_conversation_tool_capture(self) -> None:
        with self._conversation_tool_result_lock:
            self._current_conversation_tool_results = []

    def _capture_conversation_tool_result(self, result_text: str) -> None:
        with self._conversation_tool_result_lock:
            if self._current_conversation_tool_results is not None:
                self._current_conversation_tool_results.append(str(result_text or '').strip())

    def _finish_conversation_tool_capture(self) -> List[str]:
        with self._conversation_tool_result_lock:
            results = list(self._current_conversation_tool_results or [])
            self._current_conversation_tool_results = None
        return results

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
            '你要根据用户真实意图选择工具，而不是从句子里截取关键词；'
            '工具的名称、能力边界、参数和执行语义全部以当前会话提供的 tools 描述为准。'
            '需要控制机器人、查看相机、画框或抓取时，必须使用提供的工具调用，'
            '不要把工具调用写成 JSON 文本。默认用户消息不会携带相机图片。'
            '请先推理用户是在提问/确认能力/聊天，还是在要求机器人立即执行。'
            '只有用户真实意图是立即执行，并且任务属于可用工具能力时，才调用动作工具。'
            '但是如果用户问的是当前画面、现在看到什么、画面里有什么、这是什么、'
            '当前有哪些可见物体等依赖实时相机或当前环境的问题，'
            '这不是普通聊天，必须调用能查看相机或列出当前物体的工具。'
            '如果有相关工具，不要回答“我无法直接看到图像”，也不要反问用户是否要调用。'
            '如果用户要求的任务没有对应工具，不要用相似工具替代，直接说明当前能力限制。'
            '如果用户只是询问“能不能、可不可以、是什么、为什么、怎么做”等问题，'
            '优先直接回答，除非回答必须读取当前相机或工具状态。'
            '如果用户明确要求立即执行一个工具能力，按 tools 描述填写参数；'
            '如果用户一次给出多个顺序任务，必须按用户顺序逐个调用工具。'
            '如果用户说“所有/全部/每个水果”等集合任务，要循环处理每一个可见且尚未完成的目标，'
            '其中“水果/all fruits”按日常食用水果理解，不按植物学果实概念理解；'
            '辣椒、青椒、甜椒、番茄、黄瓜、茄子等日常作为蔬菜或调味食材的物体，'
            '除非用户明确点名，否则不属于“所有水果”任务目标。'
            '每完成一次抓放后继续观察/判断下一步，直到没有符合条件的目标。'
            '前一个工具成功返回后，继续判断并执行下一个尚未完成的步骤；'
            '只有所有步骤完成、遇到失败，或缺少必要信息时才停止。'
            '缺少必要参数且无法从上下文确定时，先问清楚。'
            '工具返回 success/failed/rejected/unavailable/timed out 等结果后，'
            '必须忠实转述，不要编造执行成功。'
            '你不能自己编造像素坐标、三维坐标、抓取结果或机器人状态。'
            '如果工具执行失败、拒绝、超时或不可用，只能告诉用户本次命令没有执行成功，'
            '不能自动拆分成多步继续执行。'
            '如果只是聊天或普通提问，不需要调用工具，直接简洁回答。'
        )

    def _realtime_tool_instructions_with_memory(self) -> str:
        instructions = self._omni_tool_instructions()
        context = self._conversation_context_data()
        if not context:
            return instructions
        return (
            instructions
            + '\n\n近期对话和工具结果记忆：'
            + json.dumps(context, ensure_ascii=False)
            + '\n回答“刚刚/之前/你都/已经”等追问时要优先参考这些记忆；'
            + '如果用户的新请求要求读取当前画面或执行动作，仍然按工具说明调用工具。'
        )

    def _post_tool_instructions(self, user_text: str, tool_results: List[str]) -> str:
        original_request = str(user_text or '').strip() or '当前这次用户请求'
        return (
            '刚才的工具调用已经返回。请回看原始用户请求，继续用意图而不是关键词判断下一步。'
            f'原始用户请求：{original_request}\n'
            f'刚才工具结果：{"；".join(tool_results)}\n'
            '如果原始请求只是提问、确认能力或聊天，直接回答，不要继续调用动作工具。'
            '如果原始请求要求的任务没有对应工具，不要用相似工具替代，直接说明限制。'
            '如果原始请求包含多个顺序动作，而刚才结果只是完成其中一部分，'
            '请继续调用下一个尚未完成的工具；不要因为第一步成功就提前总结。'
            '如果原始请求包含“所有/全部/每个水果”等集合任务，'
            '一次 pick-and-place 成功通常只代表完成了一个目标，必须继续处理剩余可见目标；'
            '只有工具结果已经满足原始请求的全部步骤时，才直接总结结果，不要重复调用工具。'
            '没有明确的成功工具结果时，不要说动作已经成功；'
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
        cached = self._tool_schema_cache
        if cached and time.monotonic() - self._tool_schema_cache_time < 30.0:
            return cached
        tools = self._fetch_tool_schema_from_server()
        if tools:
            self._tool_schema_cache = tools
            self._tool_schema_cache_time = time.monotonic()
            return tools
        return self._fallback_tool_schema()

    def _fetch_tool_schema_from_server(self) -> List[Dict]:
        if not self.list_tools_client.wait_for_service(timeout_sec=self.mcp_service_wait_timeout_sec):
            self.get_logger().warn(
                f'{self.list_tools_service} unavailable; continuing without MCP tools'
            )
            return []
        future = self.list_tools_client.call_async(ListTools.Request())
        response = self._wait_for_service_future(future, self.list_tools_service)
        if response is None:
            return []
        if not response.success:
            self.get_logger().warn(
                f'{self.list_tools_service} failed: {response.message}; continuing without MCP tools'
            )
            return []
        try:
            tools = json.loads(str(response.tools_json or '[]'))
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f'{self.list_tools_service} returned invalid JSON: {exc}; '
                'continuing without MCP tools'
            )
            return []
        if not isinstance(tools, list):
            self.get_logger().warn(
                f'{self.list_tools_service} did not return a tool list; continuing without MCP tools'
            )
            return []
        self._publish_status(
            f'tool_schema=server source={self.list_tools_service} count={len(tools)}'
        )
        return tools

    def _available_tool_names(self) -> List[str]:
        names = []
        for tool in self._omni_tool_schema():
            if not isinstance(tool, dict):
                continue
            function = tool.get('function', {})
            if not isinstance(function, dict):
                continue
            name = str(function.get('name') or '').strip()
            if name:
                names.append(name)
        return names

    def _tool_summaries_for_router(self) -> List[Dict]:
        summaries = []
        for tool in self._omni_tool_schema():
            if not isinstance(tool, dict):
                continue
            function = tool.get('function', {})
            if not isinstance(function, dict):
                continue
            name = str(function.get('name') or '').strip()
            if not name:
                continue
            summaries.append({
                'name': name,
                'description': str(function.get('description') or '').strip(),
                'parameters': function.get('parameters') or {},
            })
        return summaries

    def _fallback_tool_schema(self) -> List[Dict]:
        return []

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
        if not name:
            return 'call_tool failed: empty tool name'
        if not self.call_tool_client.wait_for_service(timeout_sec=self.mcp_service_wait_timeout_sec):
            return f'{self.call_tool_service} unavailable'

        request = CallTool.Request()
        request.name = name
        request.arguments_json = json.dumps(arguments, ensure_ascii=False)
        future = self.call_tool_client.call_async(request)
        response = self._wait_for_service_future(
            future,
            self.call_tool_service,
            timeout_sec=self._timeout_for_tool_call(name),
        )
        if response is None:
            return f'{self.call_tool_service} timed out'
        message = str(response.message or '').strip()
        if not message:
            status = 'success' if response.success else 'failed'
            message = f'{name} {status}'
        status = 'success' if response.success else 'failed'
        result_text = f'{name} {status}: {message}'
        result_json = str(getattr(response, 'result_json', '') or '').strip()
        memory_result_text = result_text
        if result_json and result_json != '{}':
            memory_result_text = f'{result_text}; result_json={result_json}'
        self._capture_conversation_tool_result(memory_result_text)
        self._publish_status(
            f'omni_tool_result={name} success={bool(response.success)} message={message}'
        )
        return result_text





























    def _timeout_for_tool_call(self, name: str) -> float:
        long_tools = {
            'pick_all_fruits_into_container',
            'pick_and_place_into_container',
            'pick_and_place_relative',
            'grab_api_object',
            'place_into_container',
            'place_relative_to_object',
        }
        if str(name or '') in long_tools:
            return max(float(self.mcp_tool_timeout_sec), float(self.mcp_long_tool_timeout_sec))
        return float(self.mcp_tool_timeout_sec)

    def _wait_for_service_future(self, future, label: str, timeout_sec: Optional[float] = None):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        wait_timeout = self.mcp_tool_timeout_sec if timeout_sec is None else timeout_sec
        if not event.wait(timeout=max(0.1, float(wait_timeout))):
            self.get_logger().error(f'Timed out waiting for {label}.')
            return None
        try:
            return future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed waiting for {label}: {exc}')
            return None






















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

    def _clear_realtime_buffers(self, done_event, final_text, text_parts, function_calls, lock) -> None:
        with lock:
            text_parts.clear()
        final_text['value'] = ''
        function_calls.clear()
        done_event.clear()

    def _collected_realtime_text(self, final_text, text_parts, lock) -> str:
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

    def _extract_realtime_done_text(self, response: dict) -> str:
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

    def _collect_compatible_stream_response(self, response):
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
            ' failed ',
            'failed during',
            'failed while',
            'failed moving',
            'ik failed',
            'path incomplete',
            'did not report success',
            'no scene memory',
            ' rejected:',
            ' unavailable',
            ' timed out',
            'unsupported omni tool',
        )
        return any(marker in lowered for marker in failure_markers)

    def _tool_results_complete_user_request(self, user_text: str, tool_results: List[str]) -> bool:
        if not tool_results:
            return False
        compact_request = ''.join(str(user_text or '').split())
        success_results = [
            text for text in tool_results
            if self._is_terminal_action_success_result(text)
        ]
        if not success_results:
            return False
        if self._looks_like_multi_object_task(compact_request):
            return any(self._is_pick_all_success_result(text) for text in success_results)
        if any(self._is_pick_all_success_result(text) for text in success_results):
            return True
        if any(self._is_pick_and_place_success_result(text) for text in success_results):
            return True
        if any(self._is_place_success_result(text) for text in success_results):
            return True
        if (
            any(self._is_grab_success_result(text) for text in success_results)
            and not any(token in compact_request for token in ('放', '放到', '放进', '放入'))
        ):
            return True
        return False

    @staticmethod
    def _looks_like_multi_object_task(compact_request: str) -> bool:
        text = str(compact_request or '').lower()
        collection_markers = (
            '所有',
            '全部',
            '每个',
            '每一',
            '每种',
            '都',
            'all',
            'every',
        )
        object_markers = (
            '水果',
            '物体',
            '东西',
            '目标',
            'fruit',
            'fruits',
            'object',
            'objects',
        )
        return (
            any(marker in text for marker in collection_markers)
            and any(marker in text for marker in object_markers)
        )

    def _is_terminal_action_success_result(self, result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        return (
            ' success:' in lowered
            and ' failed:' not in lowered
            and ' rejected:' not in lowered
            and ' unavailable' not in lowered
            and ' timed out' not in lowered
        )

    def _is_pick_and_place_success_result(self, result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        return (
            'pick_and_place_relative success:' in lowered
            or 'pick_and_place_into_container success:' in lowered
        )

    def _is_pick_all_success_result(self, result_text: str) -> bool:
        return 'pick_all_fruits_into_container success:' in str(result_text or '').lower()

    def _is_place_success_result(self, result_text: str) -> bool:
        lowered = str(result_text or '').lower()
        return (
            'place_relative_to_object success:' in lowered
            or 'place_into_container success:' in lowered
        )

    def _format_failed_tool_answer(self, user_text: str, failed_results: List[str]) -> str:
        return self._final_text_from_tool_results(user_text, failed_results, success=False)

    def _format_success_tool_answer(self, user_text: str, tool_results: List[str]) -> str:
        return self._final_text_from_tool_results(user_text, tool_results, success=True)

    def _final_text_from_tool_results(
        self,
        user_text: str,
        tool_results: List[str],
        success: bool,
    ) -> str:
        fallback = (
            ('任务已完成：' if success else '本次命令没有执行成功：')
            + ('；'.join(tool_results) if tool_results else ('已完成' if success else '没有工具返回结果'))
        )
        try:
            from openai import OpenAI
        except ImportError:
            return fallback
        try:
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
                            '你是机械臂任务结果播报助手。请只根据工具返回结果回答用户，'
                            '不要编造没有发生的动作，不要输出 JSON，不要复述大段内部日志。'
                            '如果结果成功，用自然中文简洁说明完成了什么；'
                            '如果失败，用自然中文说明没有成功和关键原因。'
                        ),
                    },
                    {
                        'role': 'user',
                        'content': json.dumps({
                            'original_user_request': str(user_text or ''),
                            'conversation_context': self._conversation_context_data(),
                            'tool_success': bool(success),
                            'tool_results': tool_results,
                            'output_requirement': '输出给文本框的一到两句中文自然回复',
                        }, ensure_ascii=False),
                    },
                ],
                modalities=['text'],
                stream=False,
                max_tokens=220,
            )
            text = str(response.choices[0].message.content or '').strip()
            return text or fallback
        except Exception as exc:
            self.get_logger().warn(f'Final text generation failed: {exc}')
            return fallback

    def _silence_pcm_bytes(self, byte_count: int) -> bytes:
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

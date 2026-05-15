import base64
import json
import os
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


class McpOmniClient(AudioDialogNode):
    """Qwen3.5-Omni Realtime multimodal MCP client."""

    def __init__(self) -> None:
        super().__init__('mcp_omni_client')

        self._tool_schema_cache = None
        self._tool_schema_cache_time = 0.0

        self.list_tools_client = self.create_client(ListTools, self.list_tools_service)
        self.call_tool_client = self.create_client(CallTool, self.call_tool_service)

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
        self.declare_parameter('mcp_service_wait_timeout_sec', 10.0)
        self.declare_parameter('mcp_tool_timeout_sec', 80.0)
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
        self.mcp_service_wait_timeout_sec = float(
            self.get_parameter('mcp_service_wait_timeout_sec').value
        )
        self.mcp_tool_timeout_sec = float(self.get_parameter('mcp_tool_timeout_sec').value)
        self.mcp_max_tool_rounds = max(1, int(self.get_parameter('mcp_max_tool_rounds').value))

    def _process_audio_file(self, wav_path: Path):
        self._publish_status(f'omni_audio_input={wav_path}')
        self._set_last_transcript_text('')
        self.transcript_pub.publish(String(data=''))
        response_text = self._run_omni_realtime_tool_turn(wav_path)
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
        response_text = self._run_omni_text_tool_turn(text)
        self._publish_response(response_text)
        self._speak_omni_response(response_text)
        return True, response_text

    def _text_popup_actions(self):
        return [
            ('关闭夹爪', lambda: self._run_popup_action('close_gripper', '关闭夹爪')),
            ('打开夹爪', lambda: self._run_popup_action('open_gripper', '打开夹爪')),
            ('回到初始位置', lambda: self._run_popup_action('go_home', '回到初始位置')),
        ]

    def _run_popup_action(self, tool_name: str, label: str) -> str:
        if not self._busy_lock.acquire(blocking=False):
            return '现在还在处理上一条消息，请稍等一下。'
        try:
            self._publish_status(f'text_popup_action={tool_name}')
            response_text = self._run_omni_named_tool(tool_name, {})
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
                'instructions': self._omni_tool_instructions(),
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
                    routed_call = self._route_next_text_step_with_llm(
                        route_client,
                        user_transcript['value'],
                        answer,
                        all_tool_results,
                    )
                    if routed_call:
                        if routed_call['name'] == 'final_answer':
                            final_answer = self._final_answer_from_routed_call(routed_call) or answer
                            return self._guard_final_answer(final_answer, all_tool_results)
                        self._publish_status(
                            'omni_realtime_tool_router='
                            f'{routed_call["name"]} args={routed_call["arguments"]}'
                        )
                        result_text = self._run_compatible_tool_call(routed_call)
                        all_tool_results.append(result_text)
                        if self._tool_result_failed(result_text):
                            return self._format_failed_tool_answer([result_text])
                        return self._guard_final_answer(result_text, all_tool_results)
                    return self._guard_final_answer(answer, all_tool_results)

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
        return (
            '/grab_object' in lowered
            or 'grab_object' in lowered
            or 'grab_api_object' in lowered
        )

    def _is_grab_success_result(self, result_text: str) -> bool:
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
        return user_text

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
        response = self._wait_for_service_future(future, self.call_tool_service)
        if response is None:
            return f'{self.call_tool_service} timed out'
        message = str(response.message or '').strip()
        if not message:
            status = 'success' if response.success else 'failed'
            message = f'{name} {status}'
        self._publish_status(
            f'omni_tool_result={name} success={bool(response.success)} message={message}'
        )
        return message





























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
            ' rejected:',
            ' unavailable',
            ' timed out',
            'unsupported omni tool',
        )
        return any(marker in lowered for marker in failure_markers)

    def _format_failed_tool_answer(self, failed_results: List[str]) -> str:
        return '本次命令没有执行成功：' + '；'.join(failed_results)

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

import asyncio
import math
import os
import signal
import shutil
import subprocess
import queue
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class WhisperSpeak(Node):
    """Record audio, transcribe it, and speak the text aloud."""

    def __init__(self, node_name: str = 'whisper_speak') -> None:
        super().__init__(node_name)
        self._declare_parameters()
        self._read_parameters()

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.transcript_pub = self.create_publisher(String, self.transcript_topic, 10)
        self.response_pub = self.create_publisher(String, self.response_topic, 10)
        self.service = self.create_service(Trigger, self.service_name, self.handle_record_and_speak)

        self._whisper_model = None
        self._whisper_device = ''
        self._funasr_model = None
        self._funasr_device = ''
        self._busy_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._record_process = None
        self._record_path = None
        self._record_start_time = 0.0
        self._keyboard_listener = None
        self._push_to_talk_pressed = False
        self._text_popup_lock = threading.Lock()
        self._text_popup_open = False
        self._text_popup_window = None
        self._tts_lock = threading.Lock()
        self._tts_process = None
        self._tts_generation = 0
        self._loaded_message = False
        self._auto_timer = None

        self.get_logger().info(
            'Whisper speak node ready. '
            f'service={self.service_name}, asr_engine={self.asr_engine}, '
            f'whisper_model={self.whisper_model_name}, funasr_model={self.funasr_model_name}, '
            f'record_seconds={self.record_seconds:.1f}, tts_engine={self.tts_engine}, '
            f'push_to_talk={self.push_to_talk_enabled}, '
            f'start_key={self.push_to_talk_key}, stop_key={self.stop_record_key}, '
            f'text_popup={self.text_popup_enabled}, text_key={self.text_popup_key}'
        )

        if self.push_to_talk_enabled or self.text_popup_enabled:
            self._start_keyboard_listener()

        if self.auto_run:
            self._auto_timer = self.create_timer(1.0, self._auto_run_once)

    def _declare_parameters(self) -> None:
        self.declare_parameter('service_name', '/speech/record_and_speak')
        self.declare_parameter('transcript_topic', '/speech/transcript')
        self.declare_parameter('response_topic', '/speech/response')
        self.declare_parameter('status_topic', '/speech/status')
        self.declare_parameter('record_seconds', 5.0)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('audio_device', 'default')
        self.declare_parameter('record_format', 'S16_LE')
        self.declare_parameter('output_dir', '/tmp/speech')
        self.declare_parameter('asr_engine', 'whisper')
        self.declare_parameter('whisper_model', 'small')
        self.declare_parameter('whisper_model_dir', '/home/tqq/TQQ_ws/tqq/src/speech/model')
        self.declare_parameter('whisper_language', 'zh')
        self.declare_parameter('whisper_task', 'transcribe')
        self.declare_parameter('whisper_device', '')
        self.declare_parameter(
            'funasr_model',
            '/home/tqq/TQQ_ws/tqq/src/speech/model/funasr/paraformer-zh',
        )
        self.declare_parameter('funasr_vad_model', '')
        self.declare_parameter('funasr_punc_model', '')
        self.declare_parameter('funasr_device', '')
        self.declare_parameter('funasr_hub', 'ms')
        self.declare_parameter('funasr_batch_size_s', 300.0)
        self.declare_parameter('funasr_vad_max_single_segment_time', 60000)
        self.declare_parameter('funasr_hotword', '')
        self.declare_parameter('tts_engine', 'auto')
        self.declare_parameter('tts_language', 'zh-cn')
        self.declare_parameter('tts_voice', 'kal')
        self.declare_parameter('tts_edge_voice', 'zh-CN-XiaoxiaoNeural')
        self.declare_parameter('tts_edge_rate', '+0%')
        self.declare_parameter('tts_edge_pitch', '+8Hz')
        self.declare_parameter('tts_edge_volume', '+0%')
        self.declare_parameter('tts_speed', False)
        self.declare_parameter('play_tts', True)
        self.declare_parameter('no_speech_text', '没有听到清楚的语音，请再说一次。')
        self.declare_parameter('push_to_talk_enabled', True)
        self.declare_parameter('push_to_talk_key', 'r')
        self.declare_parameter('stop_record_key', 'q')
        self.declare_parameter('min_record_seconds', 0.3)
        self.declare_parameter('text_popup_enabled', False)
        self.declare_parameter('text_popup_key', 't')
        self.declare_parameter('text_popup_title', 'MCP Text Input')
        self.declare_parameter('text_popup_prompt', '输入文本后按 Ctrl+Enter 发送，或按 Esc 取消。')
        self.declare_parameter('auto_run', False)

    def _read_parameters(self) -> None:
        self.service_name = str(self.get_parameter('service_name').value)
        self.transcript_topic = str(self.get_parameter('transcript_topic').value)
        self.response_topic = str(self.get_parameter('response_topic').value)
        self.status_topic = str(self.get_parameter('status_topic').value)
        self.record_seconds = float(self.get_parameter('record_seconds').value)
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.channels = int(self.get_parameter('channels').value)
        self.audio_device = str(self.get_parameter('audio_device').value)
        self.record_format = str(self.get_parameter('record_format').value)
        self.output_dir = Path(str(self.get_parameter('output_dir').value))
        self.asr_engine = str(self.get_parameter('asr_engine').value).strip().lower()
        self.whisper_model_name = str(self.get_parameter('whisper_model').value)
        self.whisper_model_dir = Path(str(self.get_parameter('whisper_model_dir').value)).expanduser()
        self.whisper_language = str(self.get_parameter('whisper_language').value).strip()
        self.whisper_task = str(self.get_parameter('whisper_task').value).strip()
        self.whisper_device = str(self.get_parameter('whisper_device').value).strip()
        self.funasr_model_name = str(
            Path(str(self.get_parameter('funasr_model').value).strip()).expanduser()
        )
        self.funasr_vad_model = str(self.get_parameter('funasr_vad_model').value).strip()
        self.funasr_punc_model = str(self.get_parameter('funasr_punc_model').value).strip()
        self.funasr_device = str(self.get_parameter('funasr_device').value).strip()
        self.funasr_hub = str(self.get_parameter('funasr_hub').value).strip()
        self.funasr_batch_size_s = float(self.get_parameter('funasr_batch_size_s').value)
        self.funasr_vad_max_single_segment_time = int(
            self.get_parameter('funasr_vad_max_single_segment_time').value
        )
        self.funasr_hotword = str(self.get_parameter('funasr_hotword').value).strip()
        self.tts_engine = str(self.get_parameter('tts_engine').value).strip().lower()
        self.tts_language = str(self.get_parameter('tts_language').value).strip()
        self.tts_voice = str(self.get_parameter('tts_voice').value).strip()
        self.tts_edge_voice = str(self.get_parameter('tts_edge_voice').value).strip()
        self.tts_edge_rate = str(self.get_parameter('tts_edge_rate').value).strip()
        self.tts_edge_pitch = str(self.get_parameter('tts_edge_pitch').value).strip()
        self.tts_edge_volume = str(self.get_parameter('tts_edge_volume').value).strip()
        self.tts_speed = self._as_bool(self.get_parameter('tts_speed').value)
        self.play_tts = self._as_bool(self.get_parameter('play_tts').value)
        self.no_speech_text = str(self.get_parameter('no_speech_text').value).strip()
        self.push_to_talk_enabled = self._as_bool(self.get_parameter('push_to_talk_enabled').value)
        self.push_to_talk_key = str(self.get_parameter('push_to_talk_key').value).strip().lower()
        self.stop_record_key = str(self.get_parameter('stop_record_key').value).strip().lower()
        self.min_record_seconds = float(self.get_parameter('min_record_seconds').value)
        self.text_popup_enabled = self._as_bool(self.get_parameter('text_popup_enabled').value)
        self.text_popup_key = str(self.get_parameter('text_popup_key').value).strip().lower()
        self.text_popup_title = str(self.get_parameter('text_popup_title').value).strip()
        self.text_popup_prompt = str(self.get_parameter('text_popup_prompt').value).strip()
        self.auto_run = self._as_bool(self.get_parameter('auto_run').value)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}
        return bool(value)

    def _auto_run_once(self) -> None:
        if self._loaded_message:
            return
        self._loaded_message = True
        if self._auto_timer is not None:
            self.destroy_timer(self._auto_timer)
            self._auto_timer = None
        self.get_logger().info('auto_run enabled, starting one record/transcribe/speak cycle')
        self._run_cycle()

    def handle_record_and_speak(self, request, response):
        if not self._busy_lock.acquire(blocking=False):
            response.success = False
            response.message = 'Busy running a previous cycle.'
            return response

        try:
            ok, message = self._run_cycle()
            response.success = ok
            response.message = message
        finally:
            self._busy_lock.release()
        return response

    def _run_cycle(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            wav_path = self._record_audio()
            return self._process_audio_file(wav_path)
        except Exception as exc:
            self.get_logger().error(f'Whisper speak cycle failed: {exc}')
            self._publish_status(f'error={exc}')
            return False, str(exc)

    def _record_audio(self) -> Path:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        wav_path = self.output_dir / f'record_{timestamp}.wav'

        command = ['arecord']
        if self.audio_device:
            command += ['-D', self.audio_device]
        command += [
            '-f', self.record_format,
            '-r', str(self.sample_rate),
            '-c', str(self.channels),
            '-t', 'wav',
            '-d', str(max(1, int(math.ceil(self.record_seconds)))),
            str(wav_path),
        ]

        self._publish_status(f'recording={wav_path}')
        self.get_logger().info('Recording audio: ' + ' '.join(command))
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            details = (result.stderr or result.stdout or '').strip()
            raise RuntimeError(f'arecord failed with code {result.returncode}: {details}')
        return wav_path

    def _process_audio_file(self, wav_path: Path):
        text = self._transcribe_audio(wav_path)
        return self._process_text(text, source='voice')

    def _process_text(self, text: str, source: str = 'text'):
        if not text:
            text = self.no_speech_text or '没有听到清楚的语音，请再说一次。'
            self._publish_status('transcript=')
            self.transcript_pub.publish(String(data=''))
            self._publish_response(text)
            self._speak_text_async(text)
            return True, text

        self._publish_status(f'transcript={text}')
        self.transcript_pub.publish(String(data=text))

        response_text = text
        self._publish_response(response_text)
        self._speak_text_async(response_text)
        return True, response_text

    def _start_keyboard_listener(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:
            self.get_logger().error(
                'keyboard input is enabled, but pynput is not available or cannot connect '
                'to the desktop keyboard event source. Install it with '
                f'/usr/bin/python3 -m pip install --user -U pynput. Details: {exc}'
            )
            return

        try:
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
            ready_parts = []
            if self.push_to_talk_enabled:
                ready_parts.append(
                    f'push_to_talk=ready start_key={self.push_to_talk_key} '
                    f'stop_key={self.stop_record_key}'
                )
            if self.text_popup_enabled:
                ready_parts.append(f'text_popup=ready key={self.text_popup_key}')
            self._publish_status('; '.join(ready_parts))
        except Exception as exc:
            self.get_logger().error(f'Failed to start push-to-talk keyboard listener: {exc}')

    def _key_matches_name(self, key, wanted: str) -> bool:
        wanted = wanted.lower()
        char = getattr(key, 'char', None)
        if char is not None:
            return str(char).lower() == wanted
        key_name = getattr(key, 'name', None)
        if key_name is None:
            key_name = str(key).replace('Key.', '')
        return str(key_name).lower() == wanted

    def _key_matches_push_to_talk(self, key) -> bool:
        return self._key_matches_name(key, self.push_to_talk_key)

    def _key_matches_stop_record(self, key) -> bool:
        return self._key_matches_name(key, self.stop_record_key)

    def _key_matches_text_popup(self, key) -> bool:
        return self._key_matches_name(key, self.text_popup_key)

    def _on_key_press(self, key) -> None:
        if self._text_popup_is_open():
            return

        if self.text_popup_enabled and self._key_matches_text_popup(key):
            self._open_text_popup_from_key()
            return

        if self._key_matches_stop_record(key):
            self._stop_recording_from_key()
            return

        if not self.push_to_talk_enabled or not self._key_matches_push_to_talk(key):
            return
        if self._push_to_talk_pressed:
            self.get_logger().warn('record start ignored because recording is already active.')
            return

        self._interrupt_tts_playback()
        if self._push_to_talk_pressed:
            return
        if not self._busy_lock.acquire(blocking=False):
            self.get_logger().warn('record start ignored because the node is transcribing or thinking.')
            return

        self._push_to_talk_pressed = True
        try:
            self._start_push_to_talk_recording()
        except Exception as exc:
            self._push_to_talk_pressed = False
            self._busy_lock.release()
            self.get_logger().error(f'Failed to start push-to-talk recording: {exc}')
            self._publish_status(f'error={exc}')

    def _on_key_release(self, key) -> None:
        return

    def _text_popup_is_open(self) -> bool:
        with self._text_popup_lock:
            return self._text_popup_open

    def _set_text_popup_open(self, is_open: bool) -> None:
        with self._text_popup_lock:
            self._text_popup_open = is_open

    def _open_text_popup_from_key(self) -> None:
        if self._push_to_talk_pressed:
            self.get_logger().warn('text popup ignored because recording is active.')
            return

        with self._text_popup_lock:
            if self._text_popup_open:
                window = self._text_popup_window
                if window is not None:
                    window.bring_to_front()
                return
            self._text_popup_open = True

        self._interrupt_tts_playback()

        threading.Thread(
            target=self._text_popup_worker,
            daemon=True,
        ).start()

    def _text_popup_worker(self) -> None:
        try:
            window = TextConversationWindow(
                title=self.text_popup_title or 'MCP Text Input',
                prompt=self.text_popup_prompt,
                submit_callback=self._handle_text_popup_submit,
                close_callback=lambda: self._set_text_popup_open(False),
            )
            with self._text_popup_lock:
                self._text_popup_window = window
            window.run()
        except Exception as exc:
            self.get_logger().error(f'Text popup failed: {exc}')
            self._publish_status(f'error={exc}')
        finally:
            with self._text_popup_lock:
                self._text_popup_window = None
                self._text_popup_open = False

    def _handle_text_popup_submit(self, text: str) -> str:
        if not text:
            return ''
        if not self._busy_lock.acquire(blocking=False):
            return '现在还在处理上一条消息，请稍等一下。'
        try:
            self._publish_status(f'text_popup=submitted chars={len(text)}')
            ok, response_text = self._process_text(text, source='text_popup')
            if ok:
                return response_text
            return response_text or '处理失败。'
        except Exception as exc:
            self.get_logger().error(f'Text popup submit failed: {exc}')
            self._publish_status(f'error={exc}')
            return f'处理失败：{exc}'
        finally:
            self._busy_lock.release()

    def _stop_recording_from_key(self) -> None:
        if not self._push_to_talk_pressed:
            return

        self._push_to_talk_pressed = False
        try:
            wav_path = self._stop_push_to_talk_recording()
        except Exception as exc:
            self._busy_lock.release()
            self.get_logger().error(f'Failed to stop push-to-talk recording: {exc}')
            self._publish_status(f'error={exc}')
            return

        if wav_path is None:
            self._busy_lock.release()
            return

        threading.Thread(
            target=self._finish_push_to_talk_cycle,
            args=(wav_path,),
            daemon=True,
        ).start()

    def _start_push_to_talk_recording(self) -> None:
        with self._record_lock:
            if self._record_process is not None:
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            wav_path = self.output_dir / f'ptt_{timestamp}.wav'

            command = ['arecord']
            if self.audio_device:
                command += ['-D', self.audio_device]
            command += [
                '-f', self.record_format,
                '-r', str(self.sample_rate),
                '-c', str(self.channels),
                '-t', 'wav',
                str(wav_path),
            ]

            self._record_path = wav_path
            self._record_start_time = time.monotonic()
            self._publish_status(f'push_to_talk=recording path={wav_path}')
            self.get_logger().info('Push-to-talk recording: ' + ' '.join(command))
            self._record_process = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    def _stop_push_to_talk_recording(self):
        with self._record_lock:
            process = self._record_process
            wav_path = self._record_path
            started_at = self._record_start_time
            self._record_process = None
            self._record_path = None
            self._record_start_time = 0.0

        if process is None or wav_path is None:
            return None

        duration = time.monotonic() - started_at
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()

        interrupted_by_stop = process.returncode == 1 and wav_path.exists() and wav_path.stat().st_size > 44
        if process.returncode not in (0, -signal.SIGINT) and not interrupted_by_stop:
            details = (stderr or stdout or '').strip()
            raise RuntimeError(f'arecord failed with code {process.returncode}: {details}')
        if interrupted_by_stop:
            self.get_logger().debug('arecord returned code 1 after SIGINT, but the WAV file is valid.')

        if duration < self.min_record_seconds:
            self._publish_status(f'push_to_talk=too_short duration={duration:.2f}')
            return None

        self._publish_status(f'push_to_talk=stopped duration={duration:.2f} path={wav_path}')
        return wav_path

    def _finish_push_to_talk_cycle(self, wav_path: Path) -> None:
        try:
            self._process_audio_file(wav_path)
        except Exception as exc:
            self.get_logger().error(f'Push-to-talk cycle failed: {exc}')
            self._publish_status(f'error={exc}')
        finally:
            self._busy_lock.release()

    def destroy_node(self) -> bool:
        self._interrupt_tts_playback()
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None

        with self._record_lock:
            process = self._record_process
            self._record_process = None
            self._record_path = None
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=1.0)
            except Exception:
                process.kill()

        return super().destroy_node()

    def _load_whisper_model(self):
        if self._whisper_model is not None:
            return self._whisper_model

        try:
            import torch
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                'openai-whisper is not installed. Install it with: '
                'python3 -m pip install --user -U openai-whisper'
            ) from exc

        if self.whisper_device:
            device = self.whisper_device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        download_root = None
        if str(self.whisper_model_dir):
            self.whisper_model_dir.mkdir(parents=True, exist_ok=True)
            download_root = str(self.whisper_model_dir)

        if download_root:
            self.get_logger().info(
                f'Loading Whisper model {self.whisper_model_name} on {device} '
                f'from {download_root}'
            )
            self._whisper_model = whisper.load_model(
                self.whisper_model_name,
                device=device,
                download_root=download_root,
            )
        else:
            self.get_logger().info(f'Loading Whisper model {self.whisper_model_name} on {device}')
            self._whisper_model = whisper.load_model(self.whisper_model_name, device=device)
        self._whisper_device = device
        return self._whisper_model

    def _load_funasr_model(self):
        if self._funasr_model is not None:
            return self._funasr_model

        try:
            import torch
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                'FunASR is not installed. Install it with: '
                'python3 -m pip install --user -U funasr modelscope'
            ) from exc

        if self.funasr_device:
            device = self.funasr_device
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        model_path = Path(self.funasr_model_name).expanduser()
        model_source = str(model_path) if model_path.exists() else self.funasr_model_name
        kwargs = {
            'model': model_source,
            'device': device,
        }
        if self.funasr_hub and not model_path.exists():
            kwargs['hub'] = self.funasr_hub
        if self.funasr_vad_model:
            kwargs['vad_model'] = self.funasr_vad_model
            kwargs['vad_kwargs'] = {
                'max_single_segment_time': self.funasr_vad_max_single_segment_time,
            }
        if self.funasr_punc_model:
            kwargs['punc_model'] = self.funasr_punc_model

        self.get_logger().info(
            f'Loading FunASR model {model_source} on {device} '
            f'vad={self.funasr_vad_model or "none"} punc={self.funasr_punc_model or "none"}'
        )
        self._funasr_model = AutoModel(**kwargs)
        self._funasr_device = device
        return self._funasr_model

    def _transcribe_audio(self, wav_path: Path) -> str:
        if self.asr_engine == 'funasr':
            return self._transcribe_audio_funasr(wav_path)
        if self.asr_engine != 'whisper':
            raise RuntimeError(f'Unsupported asr_engine: {self.asr_engine}')

        model = self._load_whisper_model()
        kwargs = {
            'task': self.whisper_task or 'transcribe',
            'verbose': False,
            'fp16': self._whisper_device.startswith('cuda'),
        }
        if self.whisper_language:
            kwargs['language'] = self.whisper_language

        self.get_logger().info(f'Transcribing {wav_path}')
        result = model.transcribe(str(wav_path), **kwargs)
        text = str(result.get('text', '')).strip()
        self.get_logger().info(f'Transcript ready: {len(text)} chars')
        return text

    def _transcribe_audio_funasr(self, wav_path: Path) -> str:
        model = self._load_funasr_model()
        kwargs = {
            'input': str(wav_path),
            'batch_size_s': self.funasr_batch_size_s,
        }
        if self.funasr_hotword:
            kwargs['hotword'] = self.funasr_hotword

        self.get_logger().info(f'Transcribing {wav_path} with FunASR')
        result = model.generate(**kwargs)
        text = self._extract_funasr_text(result)
        self.get_logger().info(f'Transcript ready: {len(text)} chars')
        return text

    def _extract_funasr_text(self, result) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            text = result.get('text', '')
            if text:
                return str(text).strip()
            return ''
        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, dict):
                    text = item.get('text', '')
                    if text:
                        parts.append(str(text).strip())
                elif isinstance(item, str):
                    parts.append(item.strip())
            return ''.join(parts).strip()
        return str(result).strip()

    def _speak_text_async(self, text: str) -> None:
        if not self.play_tts:
            return
        generation = self._begin_tts_generation()
        threading.Thread(
            target=self._run_tts_thread,
            args=(text, generation),
            daemon=True,
        ).start()

    def _begin_tts_generation(self) -> int:
        with self._tts_lock:
            self._tts_generation += 1
            generation = self._tts_generation
            process = self._tts_process
            self._tts_process = None
        self._terminate_process(process)
        return generation

    def _interrupt_tts_playback(self) -> None:
        with self._tts_lock:
            self._tts_generation += 1
            process = self._tts_process
            self._tts_process = None
        if process is not None:
            self._publish_status('tts=interrupted')
        self._terminate_process(process)

    def _tts_cancelled(self, generation: int) -> bool:
        with self._tts_lock:
            return generation != self._tts_generation

    def _terminate_process(self, process) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _run_tts_thread(self, text: str, generation: int) -> None:
        try:
            self._speak_text(text, generation)
        except Exception as exc:
            if not self._tts_cancelled(generation):
                self.get_logger().warn(f'TTS failed: {exc}')

    def _speak_text(self, text: str, generation: int) -> None:
        if not self.play_tts:
            return
        if self._tts_cancelled(generation):
            return

        engine = self.tts_engine or 'auto'
        if engine == 'auto':
            if self._can_import_edge_tts():
                try:
                    self._speak_with_edge_tts(text, generation)
                    return
                except Exception as exc:
                    if self._tts_cancelled(generation):
                        return
                    self.get_logger().warn(f'edge-tts failed, falling back to gTTS/flite: {exc}')
            if self._can_import_gtts():
                try:
                    self._speak_with_gtts(text, generation)
                    return
                except Exception as exc:
                    if self._tts_cancelled(generation):
                        return
                    self.get_logger().warn(f'gTTS failed, falling back to flite: {exc}')
            engine = 'flite'

        if engine == 'edge':
            self._speak_with_edge_tts(text, generation)
        elif engine == 'gtts':
            self._speak_with_gtts(text, generation)
        elif engine == 'flite':
            self._speak_with_flite(text, generation)
        elif engine == 'none':
            self.get_logger().info('TTS disabled.')
        else:
            raise RuntimeError(f'Unsupported tts_engine: {engine}')

    def _can_import_edge_tts(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    def _can_import_gtts(self) -> bool:
        try:
            import gtts  # noqa: F401
            return True
        except ImportError:
            return False

    def _speak_with_gtts(self, text: str, generation: int) -> None:
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError(
                'gTTS is not installed. Install it with: '
                'python3 -m pip install --user -U gTTS'
            ) from exc

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        mp3_path = self.output_dir / f'speech_{timestamp}.mp3'
        lang = self.tts_language.lower()
        self._publish_status(f'tts=gtts lang={lang}')
        self.get_logger().info(f'Generating gTTS audio at {mp3_path}')
        gTTS(text=text, lang=lang, slow=self.tts_speed).save(str(mp3_path))
        self._play_audio_file(mp3_path, generation)

    def _speak_with_edge_tts(self, text: str, generation: int) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                'edge-tts is not installed. Install it with: '
                'python3 -m pip install --user -U edge-tts'
            ) from exc

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        mp3_path = self.output_dir / f'speech_{timestamp}.mp3'
        voice = self.tts_edge_voice or 'zh-CN-XiaoxiaoNeural'
        self._publish_status(f'tts=edge voice={voice}')
        self.get_logger().info(f'Generating edge-tts audio at {mp3_path}')

        async def save_audio() -> None:
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=self.tts_edge_rate or '+0%',
                volume=self.tts_edge_volume or '+0%',
                pitch=self.tts_edge_pitch or '+0Hz',
            )
            await communicate.save(str(mp3_path))

        asyncio.run(save_audio())
        self._play_audio_file(mp3_path, generation)

    def _speak_with_flite(self, text: str, generation: int) -> None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        wav_path = self.output_dir / f'speech_{timestamp}.wav'
        text_path = self.output_dir / f'speech_{timestamp}.txt'
        text_path.write_text(text, encoding='utf-8')

        command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
            '-f',
            'lavfi',
            '-i',
            f'flite=textfile={text_path}:voice={self.tts_voice}',
            str(wav_path),
        ]
        self._publish_status(f'tts=flite voice={self.tts_voice}')
        self.get_logger().info('Generating flite audio: ' + ' '.join(map(str, command)))
        subprocess.run(command, check=True)
        self._play_audio_file(wav_path, generation)

    def _play_audio_file(self, audio_path: Path, generation: int) -> None:
        if not shutil.which('ffplay'):
            self.get_logger().warn(f'ffplay not found; skipping playback of {audio_path}')
            return
        if self._tts_cancelled(generation):
            return

        command = [
            'ffplay',
            '-nodisp',
            '-autoexit',
            '-loglevel',
            'error',
            str(audio_path),
        ]
        self._publish_status(f'playing={audio_path}')
        self.get_logger().info('Playing audio: ' + ' '.join(command))
        process = subprocess.Popen(command)
        with self._tts_lock:
            if generation != self._tts_generation:
                should_stop = True
            else:
                should_stop = False
                self._tts_process = process

        if should_stop:
            self._terminate_process(process)
            return

        try:
            while process.poll() is None:
                if self._tts_cancelled(generation):
                    self._terminate_process(process)
                    return
                time.sleep(0.1)
            if process.returncode != 0 and not self._tts_cancelled(generation):
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            with self._tts_lock:
                if self._tts_process is process:
                    self._tts_process = None

    def _publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _publish_response(self, text: str) -> None:
        self.response_pub.publish(String(data=text))
        self._publish_status(f'response={text}')


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
        self.history.tag_configure('user_name', foreground='#0b5cad', font=('TkDefaultFont', 10, 'bold'))
        self.history.tag_configure('assistant_name', foreground='#126b35', font=('TkDefaultFont', 10, 'bold'))
        self.history.tag_configure('system', foreground='#666666')

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

        threading.Thread(
            target=self._submit_worker,
            args=(text,),
            daemon=True,
        ).start()
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
                    self._append_message('助手', answer, name_tag='assistant_name')
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
    node = WhisperSpeak()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

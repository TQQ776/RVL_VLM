# speech

Record audio, transcribe it with Whisper or FunASR, and speak the recognized text aloud.

## Dependencies

Install the Python packages first:

```bash
/usr/bin/python3 -m pip install --user -U openai-whisper edge-tts gTTS pynput
/usr/bin/python3 -m pip install --user -U funasr modelscope
/usr/bin/python3 -m pip install --user -U torchaudio==2.10.0
```

`funasr` is optional. The default ASR engine is still Whisper, so the old flow works without changing any command.

The node also uses system tools already present on this machine:

- `arecord` for recording
- `ffmpeg` for audio synthesis playback
- `ffplay` for playback

The default TTS engine is `auto`: the node tries natural Chinese `edge-tts`, then `gTTS`,
then `ffmpeg`'s `flite` text-to-speech filter.

## Build

```bash
cd ~/TQQ_ws/tqq
colcon build --packages-select speech --symlink-install
source install/setup.bash
```

## Download Whisper model manually

The node reads Whisper model files from:

```bash
~/TQQ_ws/tqq/src/speech/model
```

Download `small.pt` manually with resume support:

```bash
mkdir -p ~/TQQ_ws/tqq/src/speech/model
cd ~/TQQ_ws/tqq/src/speech/model
wget -c -O small.pt https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
```

If `wget` is slow, download the same URL in a browser or another downloader, then put the file here:

```bash
~/TQQ_ws/tqq/src/speech/model/small.pt
```

## Download FunASR model manually

This setup uses one Chinese ASR model only. Download it into:

```bash
/home/tqq/TQQ_ws/tqq/src/speech/model/funasr/paraformer-zh
```

Manual download:

```bash
mkdir -p /home/tqq/TQQ_ws/tqq/src/speech/model/funasr
cd /home/tqq/TQQ_ws/tqq/src/speech/model/funasr
git lfs install
git clone https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git paraformer-zh
du -sh /home/tqq/TQQ_ws/tqq/src/speech/model/funasr/paraformer-zh
```

## Run

Launch the node:

```bash
ros2 launch speech whisper_speak.launch.py \
  params_file:=/home/tqq/TQQ_ws/tqq/src/speech/config/whisper_speak.yaml
```

Use push-to-talk:

```text
Press r to start recording.
Press q to stop recording, transcribe the audio, and speak the recognized text.
If the answer is still being spoken, press r to interrupt playback and start a new recording.
```

The old service trigger is still available as a fallback:

```bash
ros2 service call /speech/record_and_speak std_srvs/srv/Trigger {}
```

Useful overrides:

```bash
ros2 launch speech whisper_speak.launch.py whisper_model:=small record_seconds:=8.0 tts_engine:=gtts
ros2 launch speech whisper_speak.launch.py whisper_model:=medium
ros2 launch speech whisper_speak.launch.py asr_engine:=funasr
ros2 launch speech whisper_speak.launch.py push_to_talk_enabled:=false
ros2 launch speech whisper_speak.launch.py push_to_talk_key:=space
ros2 launch speech whisper_speak.launch.py stop_record_key:=q
ros2 launch speech whisper_speak.launch.py tts_engine:=edge tts_edge_voice:=zh-CN-XiaoxiaoNeural
ros2 launch speech whisper_speak.launch.py tts_edge_rate:=-10% tts_edge_pitch:=+10Hz
ros2 launch speech whisper_speak.launch.py tts_language:=zh-cn
ros2 launch speech whisper_speak.launch.py audio_device:=default auto_run:=true
```

Published topics:

- `/speech/transcript`
- `/speech/response`
- `/speech/status`

## Parameters

- `asr_engine`: `whisper` or `funasr`, defaults to `whisper`
- `whisper_model`: defaults to `small`
- `whisper_model_dir`: defaults to `~/TQQ_ws/tqq/src/speech/model`
- `whisper_language`: defaults to `zh`
- `funasr_model`: defaults to `/home/tqq/TQQ_ws/tqq/src/speech/model/funasr/paraformer-zh`
- `funasr_vad_model`: defaults to empty, so only one FunASR model is loaded
- `funasr_punc_model`: defaults to empty, so only one FunASR model is loaded
- `funasr_device`: empty means auto, use `cpu` or `cuda:0` to force it
- `funasr_hotword`: optional hotwords, useful for names like Franka, MoveIt, RealSense
- `push_to_talk_enabled`: defaults to `true`
- `push_to_talk_key`: defaults to `r`
- `stop_record_key`: defaults to `q`
- `min_record_seconds`: defaults to `0.3`
- `tts_engine`: `edge`, `auto`, `gtts`, `flite`, or `none`
- `tts_language`: defaults to `zh-cn`
- `tts_edge_voice`: defaults to `zh-CN-XiaoxiaoNeural`
- `tts_edge_rate`: defaults to `+0%`
- `tts_edge_pitch`: defaults to `+8Hz`
- `tts_edge_volume`: defaults to `+0%`
- `record_seconds`: how long to record
- `audio_device`: ALSA device name for `arecord`

## Notes

- Robot AI control now lives in the `mcp` package's Qwen-Omni client.
- Push-to-talk needs a desktop keyboard event source. If it cannot start, install `pynput` for `/usr/bin/python3`.
- `edge-tts` gives the most natural Chinese voice in this node, but it needs network access.
- `gTTS` is a simpler online fallback.
- `flite` is a local fallback and is mainly useful for English output.
- If you change a `.py` file or a launch file, rebuild the package.

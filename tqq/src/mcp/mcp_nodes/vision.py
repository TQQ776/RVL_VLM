from .mcp_shared import *


class VisionMixin:
    @staticmethod
    def _api_target_command_payload(
        target_name: str,
        motion_speed: Optional[float],
        request_id: str,
    ) -> str:
        target_name = str(target_name or '').strip()
        if not target_name:
            return ''
        if motion_speed is None and not request_id:
            return target_name
        payload = {'name': target_name}
        if motion_speed is not None:
            payload['motion_speed'] = motion_speed
        if request_id:
            payload['request_id'] = request_id
        return json.dumps(
            payload,
            ensure_ascii=False,
        )


    def grasp_result_callback(self, msg: String) -> None:
        try:
            payload = json.loads(str(msg.data or '{}'))
        except json.JSONDecodeError:
            payload = {'success': False, 'message': str(msg.data or '')}
        if not isinstance(payload, dict):
            payload = {'success': False, 'message': str(payload)}
        request_id = str(payload.get('request_id') or '').strip()
        if not request_id:
            return
        with self.grasp_results_lock:
            self.grasp_results[request_id] = payload
            event = self.grasp_result_events.get(request_id)
        if event is not None:
            event.set()

    def _tool_look_camera(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'look_camera failed: vision is disabled', {}
        question = str(arguments.get('question') or '').strip() or '请描述当前相机画面。'
        try:
            answer = self._ask_vision_model(question)
        except Exception as exc:
            return False, f'look_camera failed: {exc}', {}
        return True, f'look_camera success: {answer}', {'answer': answer}

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
                        {'type': 'image_url', 'image_url': {'url': image_payload['data_url']}},
                    ],
                },
            ],
            modalities=['text'],
            stream=False,
            max_tokens=self.omni_max_tokens,
        )
        return str(response.choices[0].message.content or '').strip() or '没有得到视觉回复。'

    def _ask_vision_json(self, question: str) -> Dict:
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
        )
        self._publish_status(
            f'omni_vision_json_call model={self.omni_text_model} topic={self.vision_image_topic} '
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
                        '你是机器人视觉验收器。只根据当前图像判断任务是否完成。'
                        '必须只输出严格 JSON，不要 Markdown，不要解释。'
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
            max_tokens=min(max(300, self.omni_max_tokens), 800),
            response_format={'type': 'json_object'},
        )
        return self._load_json_object(str(response.choices[0].message.content or '').strip())

    def _tool_list_api_objects(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'list_api_objects failed: vision is disabled', {}
        self._publish_api_target_command('', None)
        question = str(arguments.get('question') or '').strip()
        if not question:
            question = '请列出当前画面里清晰可见、适合机械臂抓取的物体。'
        try:
            detections = self._detect_objects_with_vision_api(question, target_name='', max_results=0)
        except Exception as exc:
            return False, f'list_api_objects failed: {exc}', {}
        if not detections:
            return True, 'list_api_objects success: 当前视觉 API 没有返回可抓取目标。', {
                'detections': [],
            }
        scene_memory = self._remember_api_detections(detections, 'list_api_objects')
        output_path = self._save_and_show_api_detection_result(
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
        return True, (
            'list_api_objects success: vision API detected: ' + ', '.join(parts)
        ), {'detections': detections, 'saved': output_path, 'scene_memory': scene_memory}

    def _tool_box_api_object(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        if not requested_name:
            return False, 'box_api_object failed: object name is empty', {}
        if not self.vision_enabled:
            return False, 'box_api_object failed: vision is disabled', {}

        is_all_objects = self._is_all_objects_box_request(requested_name)
        if is_all_objects:
            question = '请框出当前画面中所有清晰可见、适合机械臂抓取或识别的独立物体。'
            target_name = ''
        else:
            question = f'请框出画面中的“{requested_name}”。如果有多个符合的目标，请分别返回多个框。'
            target_name = requested_name

        try:
            detections = self._detect_objects_with_vision_api(
                question,
                target_name=target_name,
                max_results=0,
            )
        except Exception as exc:
            return False, f'box_api_object failed: {exc}', {}
        if not detections:
            return False, f'box_api_object failed: 视觉 API 没有找到“{requested_name}”。', {}

        scene_memory = self._remember_api_detections(
            detections,
            'box_api_object',
            aliases=[requested_name] if (not is_all_objects and len(detections) == 1) else [],
        )
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
        target_label = '所有可见物体' if is_all_objects else requested_name
        return True, (
            f'box_api_object success: 已在 Vision Box 框出 {len(detections)} 个'
            f'“{target_label}”：' + '，'.join(parts)
            + f'；saved={output_path or "disabled"}'
        ), {'detections': detections, 'saved': output_path, 'scene_memory': scene_memory}

    @staticmethod
    def _is_all_objects_box_request(text: str) -> bool:
        compact = ''.join(str(text or '').lower().split())
        if not compact:
            return False
        return compact in (
            'all',
            'allobjects',
            'everything',
            'objects',
            '所有',
            '全部',
            '所有物体',
            '全部物体',
            '所有目标',
            '全部目标',
            '所有东西',
            '全部东西',
            '可见物体',
            '画面中物体',
            '画面里的物体',
            '你看到的物体',
            '能看到的物体',
        )

    def _tool_grab_api_object(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        requested_name = str(
            arguments.get('object_name')
            or arguments.get('name')
            or arguments.get('target')
            or ''
        ).strip()
        return_home_after_grasp = self._as_bool(arguments.get('return_home_after_grasp', True))
        if not requested_name:
            return False, 'grab_api_object failed: object name is empty', {}
        if not self.vision_enabled:
            return False, 'grab_api_object failed: vision is disabled', {}
        if self.target_command_pub.get_subscription_count() < 1:
            return False, (
                'grab_api_object failed: no subscriber on '
                f'{self.target_command_topic}; start roi_economic_grasp_controller first'
            ), {}
        if self.api_detections_pub.get_subscription_count() < 1:
            return False, (
                'grab_api_object failed: no subscriber on '
                f'{self.api_detections_topic}; start roi_economic_grasp_controller first'
            ), {}

        try:
            detections = self._detect_objects_with_vision_api(
                f'请只定位最符合“{requested_name}”的一个目标。',
                target_name=requested_name,
            )
        except Exception as exc:
            return False, f'grab_api_object failed: {exc}', {}
        if not detections:
            return False, f'grab_api_object failed: 视觉 API 没有找到“{requested_name}”。', {}

        detection = detections[0]
        previous_target_entry = self._lookup_scene_object(requested_name)
        if previous_target_entry is not None:
            self._remove_world_collision_for_entry(
                previous_target_entry,
                'grab previous target world collision',
            )
        scene_memory = self._remember_api_detections(
            [detection],
            'grab_api_object',
            aliases=[requested_name],
        )
        target_name = str(detection.get('class_name', '')).strip() or requested_name
        target_entry = self._scene_entry_from_memory_result(scene_memory, target_name)
        if target_entry is not None:
            self._remove_world_collision_for_entry(target_entry, 'grab target world collision')
        request_id = f'grab_{uuid.uuid4().hex}'
        speed = self._optional_motion_speed(arguments)
        if speed is None:
            speed = self.grab_api_default_motion_speed
        self._prepare_grasp_result_wait(request_id)
        self._publish_api_target_command('', None)
        if self.api_detection_publish_settle_sec > 0.0:
            time.sleep(self.api_detection_publish_settle_sec)
        self._publish_api_target_command(target_name, speed, request_id=request_id)
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
        if not self.grab_api_wait_for_result:
            return True, f'grab_api_object success: {message}', {
                'request_id': request_id,
                'requested_name': requested_name,
                'target_name': target_name,
                'motion_speed': speed,
                'detection': detection,
                'saved': output_path,
                'scene_memory': scene_memory,
            }

        result = self._wait_for_grasp_result(request_id, self.grab_api_result_timeout_sec)
        if result is None:
            self._forget_grasp_result_wait(request_id)
            return False, (
                f'grab_api_object failed: timed out waiting {self.grab_api_result_timeout_sec:.1f}s '
                f'for EconomicGrasp result request_id={request_id}'
            ), {
                'request_id': request_id,
                'requested_name': requested_name,
                'target_name': target_name,
                'motion_speed': speed,
                'detection': detection,
                'saved': output_path,
                'scene_memory': scene_memory,
            }

        self._forget_grasp_result_wait(request_id)
        result_message = str(result.get('message') or '').strip()
        result_stage = str(result.get('stage') or '').strip()
        result_success = bool(result.get('success'))
        if not result_success:
            return False, (
                f'grab_api_object failed: EconomicGrasp result request_id={request_id} '
                f'stage={result_stage or "unknown"}: {result_message or "failed"}'
            ), {
                'request_id': request_id,
                'requested_name': requested_name,
                'target_name': target_name,
                'motion_speed': speed,
                'detection': detection,
                'saved': output_path,
                'grasp_result': result,
                'scene_memory': scene_memory,
            }

        attached = {}
        if target_entry is not None:
            attach_ok, attach_message = self._attach_held_object(target_entry)
            attached = {'success': attach_ok, 'message': attach_message, 'entry': target_entry}
            if not attach_ok:
                return False, (
                    f'grab_api_object failed after grasp: could not attach held object model: '
                    f'{attach_message}'
                ), {
                    'request_id': request_id,
                    'requested_name': requested_name,
                    'target_name': target_name,
                    'motion_speed': speed,
                    'detection': detection,
                    'saved': output_path,
                    'grasp_result': result,
                    'scene_memory': scene_memory,
                    'attached_object': attached,
                }

        success_result = {
            'request_id': request_id,
            'requested_name': requested_name,
            'target_name': target_name,
            'motion_speed': speed,
            'detection': detection,
            'saved': output_path,
            'grasp_result': result,
            'scene_memory': scene_memory,
            'attached_object': attached,
        }
        if return_home_after_grasp:
            home_ok, home_message, home_result = self._go_home_half_speed()
            success_result['return_home'] = home_result
            if not home_ok:
                return False, (
                    f'grab_api_object grasped "{target_name}" but failed returning home: '
                    f'{home_message}'
                ), success_result
            return True, (
                f'grab_api_object success: EconomicGrasp completed request_id={request_id} '
                f'target="{target_name}" stage={result_stage or "completed"}: '
                f'{result_message or "grasp completed"}; then returned home.'
            ), success_result

        return True, (
            f'grab_api_object success: EconomicGrasp completed request_id={request_id} '
            f'target="{target_name}" stage={result_stage or "completed"}: '
            f'{result_message or "grasp completed"}'
        ), success_result

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
            if max_results > 0:
                task = (
                    f'目标物体是“{target_name}”。只返回最适合抓取的一个目标框；'
                    '如果没看到这个目标，objects 返回空数组。'
                )
            else:
                task = (
                    f'目标描述是“{target_name}”。返回所有符合这个描述的独立目标框；'
                    '如果没看到符合目标，objects 返回空数组。'
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
                'stamp': {'sec': int(now.sec), 'nanosec': int(now.nanosec)},
                'frame_id': self.vision_image_topic,
            }
        return {
            'source': 'mcp_server_api',
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
        request_id: str = '',
    ) -> None:
        target_name = str(target_name or '').strip()
        if not target_name:
            self.target_command_pub.publish(String(data=''))
            return
        payload = self._api_target_command_payload(target_name, motion_speed, request_id)
        self.target_command_pub.publish(String(data=payload))

    def _prepare_grasp_result_wait(self, request_id: str) -> None:
        if not request_id:
            return
        with self.grasp_results_lock:
            self.grasp_results.pop(request_id, None)
            self.grasp_result_events[request_id] = threading.Event()

    def _wait_for_grasp_result(self, request_id: str, timeout_sec: float) -> Optional[Dict]:
        if not request_id:
            return None
        with self.grasp_results_lock:
            event = self.grasp_result_events.get(request_id)
            existing = copy.deepcopy(self.grasp_results.get(request_id))
        if existing is not None:
            return existing
        if event is None:
            event = threading.Event()
            with self.grasp_results_lock:
                self.grasp_result_events[request_id] = event
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while not event.is_set():
            if self._shutdown_requested.is_set() or self._emergency_stop_requested.is_set() or not rclpy.ok():
                self.get_logger().warn(
                    f'Interrupted while waiting for EconomicGrasp result request_id={request_id}; '
                    'stop requested.'
                )
                return {
                    'request_id': request_id,
                    'success': False,
                    'stage': 'emergency_stop',
                    'message': 'Emergency stop requested while waiting for EconomicGrasp result.',
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            event.wait(timeout=min(0.1, remaining))
        with self.grasp_results_lock:
            return copy.deepcopy(self.grasp_results.get(request_id))

    def _forget_grasp_result_wait(self, request_id: str) -> None:
        if not request_id:
            return
        with self.grasp_results_lock:
            self.grasp_result_events.pop(request_id, None)
            self.grasp_results.pop(request_id, None)

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
                    self._vision_display_status = ''

    def _vision_depth_callback(self, msg: RosImage) -> None:
        with self._depth_image_lock:
            self._latest_depth_image = msg
            self._latest_depth_image_time = time.monotonic()

    def _vision_camera_info_callback(self, msg: CameraInfo) -> None:
        with self._camera_info_lock:
            self._latest_camera_info = msg
            self._latest_camera_info_time = time.monotonic()

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
            status = ''
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
        text = VisionMixin._ascii_for_cv_text(text, 'Qwen-Omni vision')
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
            import cv2
            import numpy as np
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
